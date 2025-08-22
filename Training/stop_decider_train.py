import os
import sys
import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from peft import LoraConfig, TaskType, PeftModelForSequenceClassification, prepare_model_for_kbit_training
import wandb
from .utils import convert_chat_to_text
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import WANDB_ENTITY
from pipeline.answer_generator.prompts import gen_final_answer_prompt, gen_final_answer_docs_only_prompt


class StopDecisionDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_iterations=10, max_length=8192, target_label="prob", use_docs_only=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_label = target_label

        self.has_chat_template = (
            hasattr(self.tokenizer, 'chat_template') and
            self.tokenizer.chat_template is not None
        )
        self.use_docs_only = use_docs_only

        with open(filepath, "r", encoding="utf-8") as f:
            self.data = [json.loads(line.strip()) for line in f]
            self.data = [trace for trace in self.data if trace["iter_cnt"] < max_iterations
                         and max([trace[f"{target_label}"]] + trace[f"cont_{target_label}"] + [trace[f"last_{target_label}"]]) > 0.0]
            np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def _trace_to_encoding(self, trace):
        if self.use_docs_only:
            filtered_trace = "\n".join(f"Document: {doc['title']}: {doc['text']}" for doc in trace["history"])
            chat = gen_final_answer_docs_only_prompt(trace["question"], filtered_trace)
        else:
            filtered_trace = trace["trace"]
            chat = gen_final_answer_prompt(trace["question"], filtered_trace)

        if self.has_chat_template:  # Decoder
            encoding = self.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_dict=True
            )
        else:  # Encoder
            text = convert_chat_to_text(chat, self.tokenizer, self.use_docs_only)

            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

        encoding = {key: val.flatten() for key, val in encoding.items()}

        label1 = trace[self.target_label]
        label2 = max(trace[f"cont_{self.target_label}"] + [trace[f"last_{self.target_label}"]])
        encoding["labels"] = torch.tensor([label1, label2], dtype=torch.float)

        return encoding

    def __getitem__(self, idx):
        trace = self.data[idx]

        return self._trace_to_encoding(trace)


def compute_loss_func(target_type="abs_bce"):
    def func(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits.squeeze(-1)
        if target_type == "abs_bce":
            targets = labels[:, 0]
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        elif target_type == "abs_mse":
            targets = labels[:, 0]
            loss_fn = nn.MSELoss(reduction="sum")
        elif target_type == "soft_diff":
            targets = labels[:, 0] / (labels[:, 0] + labels[:, 1])
            targets = torch.nan_to_num(targets, nan=0.0)
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        elif target_type == "hard_diff":
            targets = (labels[:, 0] >= labels[:, 1]).float()
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            raise ValueError(f"Unknown target type: {target_type}")

        loss = loss_fn(logits, targets)
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        else:
            loss = loss / labels.numel()

        return loss

    return func


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    y_true = (labels[:, 0] >= labels[:, 1]).astype(int)
    y_score = 1 / (1 + np.exp(-logits.squeeze(-1)))
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Stop Decider Training Options")

    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID for Stop Decider")
    parser.add_argument("--train-data-path", type=str, required=True, help="Training Dataset Path")
    parser.add_argument("--eval-data-path", type=str, required=True, help="Evaluation Dataset Path")
    parser.add_argument("--target-label", type=str, default="prob", choices=["prob", "em", "f1", "acc"], help="Target label for training")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--use-docs-only", action="store_true", help="Use only documents from trace")
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
    parser.add_argument("--target-type", type=str, default="abs_bce", choices=["abs_bce", "abs_mse", "soft_diff", "hard_diff"], help="Target Type for Training")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--run-name", type=str, default=None, help="Custom WandB run name")
    parser.add_argument("--run-id", type=str, default=None, help="WandB run ID (for resuming from checkpoint)")
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
            id=args.run_id,
            resume="allow",
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
        label_names=["labels"],
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
        max_iterations=args.max_iterations,
        max_length=args.max_length,
        target_label=args.target_label,
        use_docs_only=args.use_docs_only
    )
    print(f"Number of training samples: {len(train_dataset)}", flush=True)

    eval_dataset = StopDecisionDataset(
        filepath=args.eval_data_path,
        tokenizer=tokenizer,
        max_iterations=args.max_iterations,
        max_length=args.max_length,
        target_label=args.target_label,
        use_docs_only=args.use_docs_only
    )
    print(f"Number of evaluation samples: {len(eval_dataset)}", flush=True)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_loss_func=compute_loss_func(args.target_type),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=True if args.run_id else False)

    trainer.evaluate(eval_dataset)

    print("Training completed!", flush=True)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
