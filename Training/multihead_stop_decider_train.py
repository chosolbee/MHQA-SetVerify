import os
import sys
import json
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from safetensors.torch import load_file
from peft import LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
import wandb
from .utils import convert_chat_to_text
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import WANDB_ENTITY
from pipeline.answer_generator.prompts import gen_final_answer_prompt, gen_final_answer_docs_only_prompt


class MultiheadStopDecisionDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=4096, target_label="prob", use_docs_only=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_label = target_label

        self.has_chat_template = (
            hasattr(self.tokenizer, 'chat_template') and
            self.tokenizer.chat_template is not None
        )
        self.use_docs_only = use_docs_only

        with open(filepath, "r", encoding="utf-8") as f:
            self.full_data = [json.loads(line.strip()) for line in f]
            self.data = [trace for trace in self.full_data if trace["iter_cnt"] < 10
                         and max([trace[f"{target_label}"]] + trace[f"cont_{target_label}"] + [trace[f"last_{target_label}"]]) > 0.0]
            np.random.shuffle(self.data)
            self.full_data = {trace["id"]: trace for trace in self.full_data}

    def __len__(self):
        return len(self.data)

    def _trace_to_encoding(self, trace, add_labels=True):
        if self.use_docs_only:
            filtered_trace = "\n".join(f"Document: {doc}" for doc in trace["history"])
            chat = gen_final_answer_docs_only_prompt(trace["question"], filtered_trace)
        else:
            filtered_trace = trace["trace"]
            chat = gen_final_answer_prompt(trace["question"], filtered_trace)

        if self.has_chat_template:  # Decoder
            encoding = self.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_dict=True
            )
        else:  # Encoder
            text = convert_chat_to_text(chat, self.tokenizer, self.use_docs_only)

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

        encoding = {key: val.flatten() for key, val in encoding.items()}

        if add_labels:
            encoding["curr_label"] = torch.tensor(trace[self.target_label], dtype=torch.float)
            encoding["cont_labels"] = torch.tensor(trace[f"cont_{self.target_label}"], dtype=torch.float)
            encoding["cont_mask"] = torch.tensor(trace["cont_mask"], dtype=torch.bool)
            encoding["cont_ids"] = torch.tensor(trace["cont_ids"], dtype=torch.long)
            encoding["last_label"] = torch.tensor(trace[f"last_{self.target_label}"], dtype=torch.float)

        return encoding

    def __getitem__(self, idx):
        trace = self.data[idx]

        return self._trace_to_encoding(trace, add_labels=True)

    def get_batch_from_ids(self, trace_ids, device="cpu"):
        batch = [self._trace_to_encoding(self.full_data[trace_id], add_labels=False) for trace_id in trace_ids]
        input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class MultiheadConfig(PretrainedConfig):
    model_type = "multihead"

    def __init__(
        self,
        encoder_name_or_path: str = None,
        encoder_arch: str = None,
        encoder_quantization_config: dict = None,
        encoder_lora_config: dict = None,
        dropout_prob: float = 0.1,
        use_gradient_checkpointing: bool = False,
        **kwargs,
    ):
        self.encoder_name_or_path = encoder_name_or_path
        self.encoder_arch = encoder_arch
        self.encoder_quantization_config = encoder_quantization_config
        self.encoder_lora_config = encoder_lora_config
        self.dropout_prob = dropout_prob
        self.use_gradient_checkpointing = use_gradient_checkpointing
        super().__init__(**kwargs)


class MultiheadModel(PreTrainedModel):
    config_class = MultiheadConfig
    base_model_prefix = "multihead"
    supports_gradient_checkpointing = True

    def __init__(self, config, encoder_kwargs={}, dtype=None, inference_mode=False):
        super().__init__(config)

        encoder_quantization_config = None
        if hasattr(config, "encoder_quantization_config") and config.encoder_quantization_config is not None:
            encoder_quantization_config = BitsAndBytesConfig(**config.encoder_quantization_config)

        self.encoder = AutoModel.from_pretrained(
            config.encoder_name_or_path,
            quantization_config=encoder_quantization_config,
            torch_dtype=dtype,
            **encoder_kwargs,
        )

        if hasattr(config, "encoder_lora_config") and config.encoder_lora_config is not None:
            if (
                encoder_quantization_config is not None and
                (encoder_quantization_config.load_in_4bit or encoder_quantization_config.load_in_8bit) and
                not inference_mode
            ):
                self.encoder = prepare_model_for_kbit_training(
                    self.encoder,
                    use_gradient_checkpointing=config.use_gradient_checkpointing
                )

            encoder_lora_config = LoraConfig(inference_mode=inference_mode, **config.encoder_lora_config)
            self.encoder = PeftModel(self.encoder, encoder_lora_config)

        self.classifier1 = nn.Sequential(
            nn.Dropout(config.dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size, device=self.encoder.device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.encoder.config.hidden_size, 1, device=self.encoder.device, dtype=dtype),
        )

        self.classifier2 = nn.Sequential(
            nn.Dropout(config.dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size, device=self.encoder.device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.encoder.config.hidden_size, 1, device=self.encoder.device, dtype=dtype),
        )

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = MultiheadConfig.from_pretrained(pretrained_model_path)

        model = cls(config, *model_args, **kwargs)

        state_dict_path = os.path.join(pretrained_model_path, "model.safetensors")  # assuming only one shard
        load_device = model.device.index if model.device.index is not None else "cpu"
        state_dict = load_file(state_dict_path, device=load_device)

        missing, _ = model.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            raise RuntimeError(f"Missing key(s) in state_dict: {missing}")

        return model

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict_encoder = {
            k.split(".", 1)[-1]: v
            for k, v in state_dict.items()
            if k.startswith("encoder.") and (k.endswith(".weight") or k.endswith(".bias"))
        }
        missing_encoder, unexpected_encoder = self.encoder.load_state_dict(
            state_dict_encoder, strict=strict, assign=assign
        )

        state_dict_classifier1 = {
            k.split(".", 1)[-1]: v
            for k, v in state_dict.items()
            if k.startswith("classifier1.")
        }
        missing_classifier1, unexpected_classifier1 = self.classifier1.load_state_dict(
            state_dict_classifier1, strict=strict, assign=assign
        )

        state_dict_classifier2 = {
            k.split(".", 1)[-1]: v
            for k, v in state_dict.items()
            if k.startswith("classifier2.")
        }
        missing_classifier2, unexpected_classifier2 = self.classifier2.load_state_dict(
            state_dict_classifier2, strict=strict, assign=assign
        )

        return missing_encoder + missing_classifier1 + missing_classifier2, \
               unexpected_encoder + unexpected_classifier1 + unexpected_classifier2

    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0

        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.2f}%")
        return trainable_params, all_params

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        sequence_output = outputs.last_hidden_state

        if self.config.encoder_arch == "encoder_only":
            pooled_output = sequence_output[:, 0, :]
        elif self.config.encoder_arch == "decoder_only":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).to(sequence_output.dtype)
                sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = sequence_output.mean(dim=1)
        elif self.config.encoder_arch == "encoder_decoder":
            raise NotImplementedError("Encoder-decoder architecture is not supported yet.")
        else:
            raise ValueError(f"Unsupported encoder architecture: {self.config.encoder_arch}")

        preds_head1 = self.classifier1(pooled_output)
        preds_head2 = self.classifier2(pooled_output)

        return {
            "preds_head1": preds_head1,
            "preds_head2": preds_head2,
        }

    def to(self, *args, **kwargs):
        self.encoder.to(*args, **kwargs)
        self.classifier1.to(*args, **kwargs)
        self.classifier2.to(*args, **kwargs)
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()


class MultiheadTrainer(Trainer):
    def __init__(self, *args, target_type="tdlambda", lambda_scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_type = target_type

        total_steps = self.state.max_steps
        if total_steps <= 0:
            total_steps = (len(self.get_train_dataloader()) // self.args.gradient_accumulation_steps) * self.args.num_train_epochs

        self.lambda_scheduler = lambda_scheduler
        self.lambda_scheduler.set_total_steps(total_steps)

    def _create_sliding_tensor(self, tensor, mask):
        m, n = tensor.shape
        result = torch.full((m, n, n), float("-inf"), dtype=tensor.dtype, device=tensor.device)
        for i in range(n):
            result[:, i, i + 1:] = tensor[:, :n - i - 1]
        result = result.masked_fill(~mask.unsqueeze(1).expand(m, n, n), float("-inf"))

        return result

    def _compute_mc_labels(self, inputs):
        cont_labels = inputs["cont_labels"]  # (batch_size, n)
        cont_mask = inputs["cont_mask"]  # (batch_size, n)
        masked_cont_labels = cont_labels.masked_fill(~cont_mask, float("-inf"))  # (batch_size, n)
        last_label = inputs["last_label"].unsqueeze(-1)  # (batch_size, 1)
        mc_label, _ = torch.cat([masked_cont_labels, last_label], dim=-1).max(dim=-1)  # (batch_size, )

        return mc_label

    @torch.no_grad()
    def _compute_tdlambda_labels(self, model, inputs):
        cont_labels = inputs["cont_labels"]  # (batch_size, n)
        cont_ids = inputs["cont_ids"]  # (batch_size, n)
        cont_mask = inputs["cont_mask"]  # (batch_size, n)
        valid_cont_ids = cont_ids[cont_mask]

        valid_bs_labels_head1 = torch.zeros_like(valid_cont_ids, dtype=cont_labels.dtype)
        valid_bs_labels_head2 = torch.zeros_like(valid_cont_ids, dtype=cont_labels.dtype)

        batch_size = self._train_batch_size
        for i in range(0, len(valid_cont_ids), batch_size):
            batch_cont_ids = valid_cont_ids[i:i + batch_size].tolist()
            batch_bs_inputs = self.train_dataset.get_batch_from_ids(batch_cont_ids, device=model.device)
            batch_bs_labels = model(**batch_bs_inputs)
            valid_bs_labels_head1[i:i + batch_size] = batch_bs_labels["preds_head1"].squeeze(-1)
            valid_bs_labels_head2[i:i + batch_size] = batch_bs_labels["preds_head2"].squeeze(-1)

        bs_labels_head1 = torch.zeros_like(cont_ids, dtype=cont_labels.dtype)  # (batch_size, n)
        bs_labels_head2 = torch.zeros_like(cont_ids, dtype=cont_labels.dtype)  # (batch_size, n)
        bs_labels_head1[cont_mask] = valid_bs_labels_head1
        bs_labels_head2[cont_mask] = valid_bs_labels_head2

        cont_labels = self._create_sliding_tensor(cont_labels, cont_mask)  # (batch_size, n, n)
        bs_labels_head1 = bs_labels_head1.unsqueeze(1)  # (batch_size, 1, n)
        bs_labels_head2 = bs_labels_head2.unsqueeze(1)  # (batch_size, 1, n)
        nstep_labels, _ = torch.cat([cont_labels, bs_labels_head1, bs_labels_head2], dim=1).max(dim=1)  # (batch_size, n)
        nstep_labels = nstep_labels.masked_fill(~cont_mask, 0.0)  # (batch_size, n) (unnecessary line but for clarity)

        curr_lambda = self.lambda_scheduler.get_lambda()

        n = nstep_labels.shape[-1]
        powers = torch.pow(curr_lambda, torch.arange(n, device=nstep_labels.device, dtype=nstep_labels.dtype))  # (n, )
        weighted_sum = torch.sum(nstep_labels * powers.unsqueeze(0), dim=-1)  # (batch_size, )

        mc_label = self._compute_mc_labels(inputs)  # (batch_size, )
        k = torch.sum(cont_mask, dim=-1, dtype=mc_label.dtype)  # (batch_size, )

        tdlambda_label = (1 - curr_lambda) * weighted_sum + curr_lambda ** k * mc_label  # (batch_size, )

        return tdlambda_label

    def _compute_td0_labels(self, model, inputs):
        cont_ids = inputs["cont_ids"]  # (batch_size, n)
        cont_mask = inputs["cont_mask"]  # (batch_size, n)
        next_id = cont_ids[:, 0]  # (batch_size, )
        next_mask = cont_mask[:, 0]  # (batch_size, )
        valid_next_id = next_id[next_mask]

        batch_bs_inputs = self.train_dataset.get_batch_from_ids(valid_next_id, device=model.device)
        batch_bs_labels = model(**batch_bs_inputs)
        valid_bs_labels_head1 = batch_bs_labels["preds_head1"].squeeze(-1)
        valid_bs_labels_head2 = batch_bs_labels["preds_head2"].squeeze(-1)

        bs_label_head1 = torch.zeros_like(next_id, dtype=inputs["cont_labels"].dtype)  # (batch_size, )
        bs_label_head2 = torch.zeros_like(next_id, dtype=inputs["cont_labels"].dtype)  # (batch_size, )
        bs_label_head1[next_mask] = valid_bs_labels_head1
        bs_label_head2[next_mask] = valid_bs_labels_head2

        bs_label_head1 = bs_label_head1.unsqueeze(1)  # (batch_size, 1)
        bs_label_head2 = bs_label_head2.unsqueeze(1)  # (batch_size, 1)
        td0_label, _ = torch.cat([bs_label_head1, bs_label_head2], dim=-1).max(dim=-1)  # (batch_size, )

        last_label = inputs["last_label"]  # (batch_size, )
        last_mask = cont_mask[:, 0]  # (batch_size, )
        td0_label += last_label.masked_fill(last_mask, 0.0)  # (batch_size, )

        return td0_label

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        predictions = {
            "head1": outputs["preds_head1"].squeeze(-1),
            "head2": outputs["preds_head2"].squeeze(-1),
        }

        if self.target_type == "mc":
            labels = {
                "head1": inputs["curr_label"],
                "head2": self._compute_mc_labels(inputs),
            }
        elif self.target_type == "tdlambda":
            labels = {
                "head1": inputs["curr_label"],
                "head2": self._compute_tdlambda_labels(model, inputs),
            }
        elif self.target_type == "td0":
            labels = {
                "head1": inputs["curr_label"],
                "head2": self._compute_td0_labels(model, inputs),
            }
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")

        loss = self.compute_loss_func(predictions, labels, num_items_in_batch)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            predictions = {
                "head1": outputs["preds_head1"].squeeze(-1),
                "head2": outputs["preds_head2"].squeeze(-1),
            }

            labels = {
                "head1": inputs["curr_label"],
                "head2": self._compute_mc_labels(inputs),
            }

            loss = self.compute_loss_func(predictions, labels)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, predictions, labels)


class LambdaScheduler:
    def __init__(self, lambda_init=1.0, lambda_final=0.1, scheduler_type="none"):
        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.scheduler_type = scheduler_type

        self.total_steps = None
        self.current_step = 0

    def set_total_steps(self, total_steps):
        self.total_steps = total_steps

    def step(self):
        self.current_step += 1

    def get_lambda(self):
        if self.total_steps is None:
            return self.lambda_init

        progress = min(self.current_step / self.total_steps, 1.0)

        if self.scheduler_type == "linear":
            return self.lambda_init - progress * (self.lambda_init - self.lambda_final)
        elif self.scheduler_type == "exponential":
            return self.lambda_init * (self.lambda_final / self.lambda_init) ** progress
        elif self.scheduler_type == "cosine":
            return self.lambda_final + 0.5 * (self.lambda_init - self.lambda_final) * (1 + np.cos(np.pi * progress))
        elif self.scheduler_type == "none":
            return self.lambda_init
        else:
            raise ValueError(f"Unsupported schedule type: {self.scheduler_type}")


class LambdaDecayCallback(TrainerCallback):
    def __init__(self, lambda_scheduler):
        self.lambda_scheduler = lambda_scheduler

    def on_step_end(self, args, state, control, **kwargs):
        self.lambda_scheduler.step()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if wandb.run is not None:
            wandb.log({"train/lambda": self.lambda_scheduler.get_lambda()})


def compute_loss_func(predictions, labels, num_items_in_batch=None):
    loss_fn = nn.MSELoss(reduction="sum")
    loss_head1 = loss_fn(predictions["head1"], labels["head1"])
    loss_head2 = loss_fn(predictions["head2"], labels["head2"])
    loss = loss_head1 + loss_head2
    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch
    else:
        loss = loss / labels["head1"].numel()

    return loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    pred_head1 = predictions["head1"]
    pred_head2 = predictions["head2"]
    labels_head1 = labels["head1"]
    labels_head2 = labels["head2"]

    y_true = (labels_head1 >= labels_head2).astype(int)
    y_score = np.clip(pred_head1 - pred_head2 + 0.5, 0.0, 1.0)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Stop Decider Training Options")

    parser.add_argument("--model-id", type=str, default="microsoft/deberta-v3-large", help="Model ID for Stop Decider")
    parser.add_argument("--model-arch", type=str, default="encoder_only", choices=["encoder_only", "decoder_only", "encoder_decoder"], help="Model Architecture")
    parser.add_argument("--train-data-path", type=str, required=True, help="Training Dataset Path")
    parser.add_argument("--eval-data-path", type=str, required=True, help="Evaluation Dataset Path")
    parser.add_argument("--target-label", type=str, default="prob", choices=["prob", "em", "f1"], help="Target label for training")
    parser.add_argument("--lambda-init", type=float, default=1.0, help="Initial Value of lambda for TD Lambda")
    parser.add_argument("--lambda-final", type=float, default=0.1, help="Final Value of lambda for TD Lambda")
    parser.add_argument("--lambda-scheduler-type", type=str, default="cosine", choices=["linear", "exponential", "cosine", "none"], help="TD Lambda Scheduler Type")
    parser.add_argument("--use-docs-only", action="store_true", help="Use only documents from trace")
    parser.add_argument("--dropout-prob", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization for training (QLoRA)")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for training")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA Rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.01, help="LoRA Dropout")
    parser.add_argument("--lora-bias", type=str, default="none", choices=["none", "all", "lora_only"], help="LoRA Bias Type")
    parser.add_argument("--trainer-output-dir", type=str, help="Training Output Path")
    parser.add_argument("--max-length", type=int, default=2048, help="Max Length of Inputs")
    parser.add_argument("--optimizer", type=str, default="adamw_torch_fused", help="Optimizer Type")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning Rate")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", help="Learning Rate Scheduler Type")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup Ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight Decay")
    parser.add_argument("--max-grad-norm", type=float, default=30.0, help="Max Gradient Norm")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch Size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32, help="Gradient Accumulation Steps")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Use Gradient Checkpointing")
    parser.add_argument("--bf16", action="store_true", help="Use BF16")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of Epochs")
    parser.add_argument("--eval-steps", type=int, default=200, help="Evaluation Steps")
    parser.add_argument("--save-steps", type=int, default=200, help="Save Steps")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Total Number of Saved Checkpoints")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging Steps")
    parser.add_argument("--deepspeed-config", type=str, default="Training/deepspeed_config.json", help="DeepSpeed Configuration File Path")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true", help="Find unused parameters in DDP")
    parser.add_argument("--target-type", type=str, default="tdlambda", choices=["mc", "tdlambda", "td0"], help="Target Type for Training")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--run-name", type=str, default=None, help="Custom WandB run name")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")

    args = parser.parse_args()

    return args


def main(args):
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if local_rank == 0 and not args.disable_wandb:
        wandb.init(
            project="stop-decider-train",
            entity=WANDB_ENTITY,
            name=args.run_name,
            config=args,
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"

    config = MultiheadConfig(
        encoder_name_or_path=args.model_id,
        encoder_arch=args.model_arch,
        encoder_quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16" if args.bf16 else "float32",
        } if args.use_4bit else None,
        encoder_lora_config={
            "task_type": TaskType.SEQ_CLS,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "r": args.lora_r,
            "bias": args.lora_bias,
            "target_modules": "all-linear",
        } if args.use_lora else None,
        dropout_prob=args.dropout_prob,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    model = MultiheadModel(
        config,
        encoder_kwargs={
            "device_map": {"": local_rank},
        },
        inference_mode=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Model and tokenizer loaded successfully.", flush=True)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.trainer_output_dir,
        optim=args.optimizer,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        num_train_epochs=args.num_epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        label_names=["curr_label", "cont_labels", "cont_mask", "cont_ids", "last_label"],
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        report_to=None if args.disable_wandb else ["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
    )

    train_dataset = MultiheadStopDecisionDataset(args.train_data_path, tokenizer, args.max_length, args.target_label, args.use_docs_only)
    print(f"Number of training samples: {len(train_dataset)}", flush=True)

    eval_dataset = MultiheadStopDecisionDataset(args.eval_data_path, tokenizer, args.max_length, args.target_label, args.use_docs_only)
    print(f"Number of evaluation samples: {len(eval_dataset)}", flush=True)

    lambda_scheduler = LambdaScheduler(
        lambda_init=args.lambda_init,
        lambda_final= args.lambda_final,
        scheduler_type=args.lambda_scheduler_type,
    )

    lambda_decay_callback = LambdaDecayCallback(lambda_scheduler)

    trainer = MultiheadTrainer(
        target_type=args.target_type,
        lambda_scheduler=lambda_scheduler,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_loss_func=compute_loss_func,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[lambda_decay_callback],
    )

    trainer.train()

    trainer.evaluate(eval_dataset)

    print("Training completed!", flush=True)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
