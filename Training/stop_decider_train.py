import os
import sys
import json
import argparse
import itertools
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, TaskType, PeftModelForSequenceClassification, prepare_model_for_kbit_training
import wandb
from .utils import extract_documents_only, convert_chat_to_text
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import WANDB_ENTITY
from pipeline.answer_generator.prompts import gen_final_answer_prompt, gen_final_answer_docs_only_prompt


class StopDecisionDataset(Dataset):
    def __init__(
        self,
        filepath,
        tokenizer,
        max_items=None,
        max_length=4096,
        target_label="prob",
        use_docs_only=False,
        pairwise=False,
        pairwise_threshold=0.3
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_label = target_label

        self.has_chat_template = (
            hasattr(self.tokenizer, 'chat_template') and
            self.tokenizer.chat_template is not None
        )
        self.use_docs_only = use_docs_only
        self.pairwise = pairwise

        with open(filepath, "r", encoding="utf-8") as f:
            self.data = [json.loads(line.strip()) for line in f]
            self.data = [trace for trace in self.data if trace["iter_cnt"] < 10]

        if pairwise:
            pos_samples = [trace for trace in self.data if trace[target_label] - trace[f"max_cont_{target_label}"] > pairwise_threshold]
            neg_samples = [trace for trace in self.data if trace[f"max_cont_{target_label}"] - trace[target_label] > pairwise_threshold]
            if max_items is None:
                self.data = list(itertools.product(pos_samples, neg_samples))
            else:
                seen_indices = set()
                self.data = []
                while len(self.data) < max_items:
                    pos_idx = np.random.choice(len(pos_samples))
                    neg_idx = np.random.choice(len(neg_samples))
                    if (pos_idx, neg_idx) not in seen_indices:
                        seen_indices.add((pos_idx, neg_idx))
                        self.data.append((pos_samples[pos_idx], neg_samples[neg_idx]))
        else:
            self.data = [trace for trace in self.data if not (trace[target_label] == 0.0 and trace[f"max_cont_{target_label}"] == 0.0)]
            np.random.shuffle(self.data)
            if max_items is not None:
                self.data = self.data[:max_items]

    def __len__(self):
        return len(self.data)

    def _trace_to_encoding(self, trace, add_labels=True, prefix=None):
        if self.use_docs_only:
            filtered_trace = extract_documents_only(trace["trace"])
            chat = gen_final_answer_docs_only_prompt(trace["question"], filtered_trace)
        else:
            filtered_trace = trace["trace"]
            chat = gen_final_answer_prompt(trace["question"], filtered_trace)

        label1 = trace[self.target_label]
        label2 = trace[f"max_cont_{self.target_label}"]

        if self.has_chat_template: # Decoder
            encoding = self.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_dict=True
            )
        else: # Encoder
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
            encoding["labels"] = torch.tensor([label1, label2], dtype=torch.float)
        if prefix is not None:
            encoding = {f"{prefix}_{key}": val for key, val in encoding.items()}

        return encoding

    def __getitem__(self, idx):
        trace = self.data[idx]

        if self.pairwise:
            pos_trace, neg_trace = trace
            pos_encoding = self._trace_to_encoding(pos_trace, add_labels=False, prefix="pos")
            neg_encoding = self._trace_to_encoding(neg_trace, add_labels=False, prefix="neg")

            return {**pos_encoding, **neg_encoding}
        else:
            encoding = self._trace_to_encoding(trace, add_labels=True)

            return encoding


class StopDecisionTrainer(Trainer):
    def __init__(self, pairwise=False, target_type="abs_bce", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairwise = pairwise
        self.target_type = target_type

    def compute_loss_func_pairwise(self, pos_outputs, neg_outputs, num_items_in_batch=None):
        pos_logits = pos_outputs.logits.squeeze(-1)
        neg_logits = neg_outputs.logits.squeeze(-1)

        if self.target_type == "margin":
            targets = torch.ones_like(pos_logits)
            loss = nn.MarginRankingLoss(margin=1.0, reduction="sum")(pos_logits, neg_logits, targets)
        elif self.target_type == "ranknet":
            logits = torch.sigmoid(pos_logits - neg_logits)
            targets = torch.ones_like(pos_logits)
            loss = nn.BCELoss(reduction="sum")(logits, targets)
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        else:
            loss = loss / pos_logits.numel()

        return loss

    def compute_loss_func_pointwise(self, outputs, labels, num_items_in_batch=None):
        logits = outputs.logits.squeeze(-1)

        if self.target_type == "abs_bce":
            targets = labels[:, 0]
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        elif self.target_type == "abs_mse":
            targets = labels[:, 0]
            loss_fn = nn.MSELoss(reduction="sum")
        elif self.target_type == "soft_diff":
            targets = labels[:, 0] / (labels[:, 0] + labels[:, 1])
            targets = torch.nan_to_num(targets, nan=0.0)
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        elif self.target_type == "hard_diff":
            targets = (labels[:, 0] >= labels[:, 1]).float()
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

        loss = loss_fn(logits, targets)
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        else:
            loss = loss / labels.numel()

        return loss

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        if self.pairwise:
            pos_input_ids = inputs["pos_input_ids"]
            pos_attention_mask = inputs["pos_attention_mask"]
            pos_outputs = model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
            )

            neg_input_ids = inputs["neg_input_ids"]
            neg_attention_mask = inputs["neg_attention_mask"]
            neg_outputs = model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
            )

            outputs = {
                "pos": pos_outputs,
                "neg": neg_outputs,
            }
            loss = self.compute_loss_func_pairwise(pos_outputs, neg_outputs, num_items_in_batch)
        else:
            outputs = model(**inputs)
            labels = inputs["labels"]
            loss = self.compute_loss_func_pointwise(outputs, labels, num_items_in_batch)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            if self.pairwise:
                predictions = outputs["pos"].logits - outputs["neg"].logits
                labels = torch.ones_like(predictions, dtype=torch.float)
            else:
                predictions = outputs.logits
                if self.target_type == "abs_bce":
                    labels = inputs["labels"][:, 0]
                elif self.target_type == "abs_mse":
                    labels = inputs["labels"][:, 0]
                elif self.target_type == "soft_diff":
                    labels = inputs["labels"][:, 0] / (inputs["labels"][:, 0] + inputs["labels"][:, 1])
                    labels = torch.nan_to_num(labels, nan=0.0)
                elif self.target_type == "hard_diff":
                    labels = (inputs["labels"][:, 0] >= inputs["labels"][:, 1]).float()
                else:
                    raise ValueError(f"Unknown target type: {self.target_type}")

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, predictions, labels)


def collate_fn(pairwise=False):
    def func_pairwise(batch):
        pos_input_ids = torch.stack([item["pos_input_ids"] for item in batch])
        pos_attention_mask = torch.stack([item["pos_attention_mask"] for item in batch])
        neg_input_ids = torch.stack([item["neg_input_ids"] for item in batch])
        neg_attention_mask = torch.stack([item["neg_attention_mask"] for item in batch])

        return {
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_attention_mask,
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_attention_mask,
        }

    def func_pointwise(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return func_pairwise if pairwise else func_pointwise


def compute_metrics(pairwise=False, decision_threshold=0.8):
    def func(eval_pred):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        logits, labels = eval_pred
        preds = sigmoid(logits.squeeze(-1))

        y_pred = (preds > decision_threshold).astype(int)
        y_true = (labels[:, 0] >= labels[:, 1]).astype(int)
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

    return None if pairwise else func


def parse_args():
    parser = argparse.ArgumentParser(description="Stop Decider Training Options")

    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID for Stop Decider")
    parser.add_argument("--train-data-path", type=str, required=True, help="Training Dataset Path")
    parser.add_argument("--eval-data-path", type=str, required=True, help="Evaluation Dataset Path")
    parser.add_argument("--test-data-path", type=str, required=True, help="Test Dataset Path")
    parser.add_argument("--max-train-items", type=int, help="Maximum number of training items")
    parser.add_argument("--max-eval-items", type=int, help="Maximum number of evaluation items")
    parser.add_argument("--max-test-items", type=int, help="Maximum number of test items")
    parser.add_argument("--target-label", type=str, default="prob", choices=["prob", "em", "f1"], help="Target label for training")
    parser.add_argument("--use-docs-only", action="store_true", help="Use only documents from trace")
    parser.add_argument("--pairwise", action="store_true", help="Use pairwise training")
    parser.add_argument("--pairwise-threshold", type=float, default=0.3, help="Pairwise training threshold")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization for training (QLoRA)")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for training")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA Rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.01, help="LoRA Dropout")
    parser.add_argument("--lora-bias", type=str, default="none", choices=["none", "all", "lora_only"], help="LoRA Bias Type")
    parser.add_argument("--trainer-output-dir", type=str, help="Training Output Path")
    parser.add_argument("--max-length", type=int, default=4096, help="Max Length of Inputs")
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
    parser.add_argument("--decision-threshold", type=float, default=0.8, help="Threshold for stop decision")
    parser.add_argument("--target-type", type=str, default="abs_bce", choices=["abs_bce", "abs_mse", "soft_diff", "hard_diff", "margin", "ranknet"], help="Target Type for Training")
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

    nf4_config = None
    if args.use_4bit:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        quantization_config=nf4_config,
        device_map={"": local_rank},
        num_labels=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Model and tokenizer loaded successfully.", flush=True)

    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias=args.lora_bias,
            target_modules="all-linear",
        )

        if args.use_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        model = PeftModelForSequenceClassification(model, lora_config)

        print("LoRA configuration applied successfully.", flush=True)

    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

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
        label_names=["pos_input_ids", "pos_attention_mask", "neg_input_ids", "neg_attention_mask"] if args.pairwise else ["labels"],
        metric_for_best_model="eval_loss",
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        report_to=None if args.disable_wandb else ["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
    )

    train_dataset = StopDecisionDataset(
        filepath=args.train_data_path,
        tokenizer=tokenizer,
        max_items=args.max_train_items,
        max_length=args.max_length,
        target_label=args.target_label,
        use_docs_only=args.use_docs_only,
        pairwise=args.pairwise,
        pairwise_threshold=args.pairwise_threshold,
    )
    print(f"Number of training samples: {len(train_dataset)}", flush=True)

    eval_dataset = StopDecisionDataset(
        filepath=args.eval_data_path,
        tokenizer=tokenizer,
        max_items=args.max_eval_items,
        max_length=args.max_length,
        target_label=args.target_label,
        use_docs_only=args.use_docs_only,
        pairwise=args.pairwise,
        pairwise_threshold=args.pairwise_threshold,
    )
    print(f"Number of evaluation samples: {len(eval_dataset)}", flush=True)

    trainer = StopDecisionTrainer(
        pairwise=args.pairwise,
        target_type=args.target_type,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics(args.pairwise, args.decision_threshold),
        data_collator=collate_fn(args.pairwise),
        processing_class=tokenizer,
    )

    trainer.train()

    print("Training completed. Evaluating on test dataset...\n", flush=True)

    test_dataset = StopDecisionDataset(
        filepath=args.test_data_path,
        tokenizer=tokenizer,
        max_items=args.max_test_items,
        max_length=args.max_length,
        target_label=args.target_label,
        use_docs_only=args.use_docs_only,
        pairwise=args.pairwise,
        pairwise_threshold=args.pairwise_threshold,
    )
    print(f"Number of test samples: {len(test_dataset)}", flush=True)

    _, _, metrics = trainer.predict(test_dataset)

    print("Test Metrics:", flush=True)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
