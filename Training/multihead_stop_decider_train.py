import os
import sys
import json
import argparse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
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
    set_seed,
)
from safetensors.torch import load_file
from peft import LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
import wandb
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import WANDB_ENTITY
from pipeline.answer_generator.prompts import gen_final_answer_prompt


class MultiheadStopDecisionDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=4096, target_label="prob"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_label = target_label

        with open(filepath, "r", encoding="utf-8") as f:
            self.data = [json.loads(line.strip()) for line in f]
            self.data = [trace for trace in self.data if trace["iter_cnt"] < 10]
            np.random.shuffle(self.data)

        pos_samples = [trace for trace in self.data if trace[target_label] > trace[f"max_cont_{target_label}"]]
        neu_samples = [trace for trace in self.data if trace[target_label] == trace[f"max_cont_{target_label}"]]
        neg_samples = [trace for trace in self.data if trace[target_label] < trace[f"max_cont_{target_label}"]]
        min_len = min(len(pos_samples), len(neu_samples), len(neg_samples))
        self.data = pos_samples[:min_len] + neu_samples[:min_len] + neg_samples[:min_len]
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trace = self.data[idx]
        chat = gen_final_answer_prompt(trace["question"], trace["trace"])
        label1 = trace[self.target_label]
        label2 = trace[f"max_cont_{self.target_label}"]

        encoding = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_dict=True
        )

        encoding = {key: val.flatten() for key, val in encoding.items()}
        encoding["labels"] = torch.tensor([label1, label2], dtype=torch.float)

        return encoding


class MultiheadConfig(PretrainedConfig):
    model_type = "multihead"

    def __init__(
        self,
        encoder_name_or_path: str = None,
        encoder_quantization_config: dict = None,
        encoder_lora_config: dict = None,
        dropout_prob: float = 0.1,
        use_gradient_checkpointing: bool = False,
        **kwargs,
    ):
        self.encoder_name_or_path = encoder_name_or_path
        self.encoder_quantization_config = encoder_quantization_config
        self.encoder_lora_config = encoder_lora_config
        self.dropout_prob = dropout_prob
        self.use_gradient_checkpointing = use_gradient_checkpointing
        super().__init__(**kwargs)


class MultiheadModel(PreTrainedModel):
    config_class = MultiheadConfig
    base_model_prefix = "multihead"

    def __init__(self, config, encoder_kwargs={}, dtype=None, inference_mode=False):
        super().__init__(config)

        encoder_quantization_config = None
        if hasattr(config, "encoder_quantization_config"):
            encoder_quantization_config = BitsAndBytesConfig(**config.encoder_quantization_config)

        self.encoder = AutoModel.from_pretrained(
            config.encoder_name_or_path,
            quantization_config=encoder_quantization_config,
            torch_dtype=dtype,
            **encoder_kwargs,
        )

        if (encoder_quantization_config.load_in_4bit or encoder_quantization_config.load_in_8bit) and not inference_mode:
            self.encoder = prepare_model_for_kbit_training(self.encoder, use_gradient_checkpointing=config.use_gradient_checkpointing)

        if hasattr(config, "encoder_lora_config"):
            encoder_lora_config = LoraConfig(inference_mode=inference_mode, **config.encoder_lora_config)
            self.encoder = PeftModel(self.encoder, encoder_lora_config)

        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier_head1 = nn.Linear(self.encoder.config.hidden_size, 1, device=self.encoder.device, dtype=dtype)
        self.classifier_head2 = nn.Linear(self.encoder.config.hidden_size, 1, device=self.encoder.device, dtype=dtype)

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
            if k.startswith("encoder.") and k.endswith(".weight")
        }
        missing_encoder, unexpected_encoder = self.encoder.load_state_dict(
            state_dict_encoder, strict=strict, assign=assign
        )

        state_dict_classifier_head1 = {
            k.split(".", 1)[-1]: v
            for k, v in state_dict.items()
            if k.startswith("classifier_head1.")
        }
        missing_classifier_head1, unexpected_classifier_head1 = self.classifier_head1.load_state_dict(
            state_dict_classifier_head1, strict=strict, assign=assign
        )

        state_dict_classifier_head2 = {
            k.split(".", 1)[-1]: v
            for k, v in state_dict.items()
            if k.startswith("classifier_head2.")
        }
        missing_classifier_head2, unexpected_classifier_head2 = self.classifier_head2.load_state_dict(
            state_dict_classifier_head2, strict=strict, assign=assign
        )

        return missing_encoder + missing_classifier_head1 + missing_classifier_head2, \
               unexpected_encoder + unexpected_classifier_head1 + unexpected_classifier_head2

    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0

        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.2f}%")
        return trainable_params, all_params

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        sequence_output = outputs.last_hidden_state

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).to(sequence_output.dtype)
            sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = sequence_output.mean(dim=1)

        pooled_output = self.dropout(pooled_output)

        preds_head1 = self.classifier_head1(pooled_output)
        preds_head2 = self.classifier_head2(pooled_output)

        return {
            "preds_head1": preds_head1,
            "preds_head2": preds_head2,
        }

    def to(self, *args, **kwargs):
        self.encoder.to(*args, **kwargs)
        self.dropout.to(*args, **kwargs)
        self.classifier_head1.to(*args, **kwargs)
        self.classifier_head2.to(*args, **kwargs)
        return self


class MultiheadTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = self.compute_loss_func(outputs, inputs, num_items_in_batch)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            if prediction_loss_only:
                return (loss, None, None)

            predictions = {
                "head1": outputs["preds_head1"].squeeze(-1),
                "head2": outputs["preds_head2"].squeeze(-1),
            }

            labels = {
                "head1": inputs["labels"][:, 0],
                "head2": inputs["labels"][:, 1],
            }

        return (loss, predictions, labels)


def compute_loss_func(outputs, inputs, num_items_in_batch=None):
    loss_fn = nn.MSELoss(reduction="sum")
    loss_head1 = loss_fn(outputs["preds_head1"].squeeze(-1), inputs["labels"][:, 0])
    loss_head2 = loss_fn(outputs["preds_head2"].squeeze(-1), inputs["labels"][:, 1])
    loss = loss_head1 + loss_head2
    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch
    else:
        loss = loss / inputs["labels"].numel()

    return loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    pred_head1 = predictions["head1"]
    pred_head2 = predictions["head2"]
    labels_head1 = labels["head1"]
    labels_head2 = labels["head2"]

    y_pred = (pred_head1 > pred_head2).astype(int)
    y_true = (labels_head1 > labels_head2).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Stop Decider Training Options")

    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID for Stop Decider")
    parser.add_argument("--train-data-path", type=str, required=True, help="Training Dataset Path")
    parser.add_argument("--eval-data-path", type=str, required=True, help="Evaluation Dataset Path")
    parser.add_argument("--test-data-path", type=str, required=True, help="Test Dataset Path")
    parser.add_argument("--target-label", type=str, default="prob", choices=["prob", "em", "f1"], help="Target label for training")
    parser.add_argument("--dropout-prob", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA Rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.01, help="LoRA Dropout")
    parser.add_argument("--lora-bias", type=str, default="none", choices=["none", "all", "lora_only"], help="LoRA Bias Type")
    parser.add_argument("--trainer-output-dir", type=str, help="Training Output Path")
    parser.add_argument("--max-length", type=int, default=4096, help="Max Length of Tokenizer")
    parser.add_argument("--optimizer", type=str, default="adamw_torch_fused", help="Optimizer Type")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning Rate")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", help="Learning Rate Scheduler Type")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup Ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight Decay")
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
        encoder_quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16" if args.bf16 else "float32",
        },
        encoder_lora_config={
            "task_type": TaskType.SEQ_CLS,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "r": args.lora_r,
            "bias": args.lora_bias,
            "target_modules": "all-linear",
        },
        dropout_prob=args.dropout_prob,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    model = MultiheadModel(
        config,
        encoder_kwargs={
            "device_map": {"": local_rank},
            "use_cache": False,
        },
        inference_mode=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Model and tokenizer loaded successfully.", flush=True)

    model.print_trainable_parameters()

    print("LoRA configuration applied successfully.", flush=True)

    training_args = TrainingArguments(
        output_dir=args.trainer_output_dir,
        optim=args.optimizer,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
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
        load_best_model_at_end=True,
        label_names=["labels"],
        metric_for_best_model="eval_loss",
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        report_to=None if args.disable_wandb else ["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
    )

    train_dataset = MultiheadStopDecisionDataset(args.train_data_path, tokenizer, args.max_length, args.target_label)
    print(f"Number of training samples: {len(train_dataset)}", flush=True)

    eval_dataset = MultiheadStopDecisionDataset(args.eval_data_path, tokenizer, args.max_length, args.target_label)
    print(f"Number of evaluation samples: {len(eval_dataset)}", flush=True)

    trainer = MultiheadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_loss_func=compute_loss_func,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()

    print("Training completed. Evaluating on test dataset...\n", flush=True)

    test_dataset = MultiheadStopDecisionDataset(args.test_data_path, tokenizer, args.max_length, args.target_label)
    print(f"Number of test samples: {len(test_dataset)}", flush=True)

    _, _, metrics = trainer.predict(test_dataset)

    print("Test Metrics:", flush=True)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
