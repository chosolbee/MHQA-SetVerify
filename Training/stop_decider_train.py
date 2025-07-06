import os
import sys
import json
import argparse
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
from peft import LoraConfig, TaskType, PeftModel
import wandb
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import WANDB_ENTITY
from pipeline.answer_generator.prompts import gen_final_answer_prompt


class StopDecisionDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=8192):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(filepath, "r", encoding="utf-8") as f:
            self.data = [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trace = self.data[idx]
        chat = gen_final_answer_prompt(trace["question"], trace["trace"])
        label = trace["prob"]

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
        encoding["labels"] = torch.tensor(label, dtype=torch.float)

        return encoding


def compute_loss_func(outputs, labels, num_items_in_batch=None):
    logits = outputs.logits.squeeze(-1)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    loss = loss_fn(logits, labels)

    return loss


def compute_metrics(eval_pred, threshold):
    logits, labels = eval_pred
    preds = 1 / (1 + np.exp(-(logits.squeeze(-1))))

    y_pred = (preds > threshold).astype(int)
    y_true = (labels > threshold).astype(int)
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
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of Epochs")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation Steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save Steps")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Total Number of Saved Checkpoints")
    parser.add_argument("--logging-steps", type=int, default=100, help="Logging Steps")
    parser.add_argument("--deepspeed-config", type=str, default="Training/deepspeed_config.json", help="DeepSpeed Configuration File Path")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true", help="Find unused parameters in DDP")
    parser.add_argument("--threshold", type=float, default=0.8, help="Threshold for stop decision")
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

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        quantization_config=nf4_config,
        use_cache=False,
        device_map={"": local_rank},
        num_labels=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Model and tokenizer loaded successfully.", flush=True)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias=args.lora_bias,
        target_modules="all-linear",
        modules_to_save=["score"],
    )

    model = PeftModel(model, lora_config)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    if args.fp16:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
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
        fp16=args.fp16,
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

    train_dataset = StopDecisionDataset(args.train_data_path, tokenizer, args.max_length)
    print(f"Number of training samples: {len(train_dataset)}", flush=True)

    eval_dataset = StopDecisionDataset(args.eval_data_path, tokenizer, args.max_length)
    print(f"Number of evaluation samples: {len(eval_dataset)}", flush=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_loss_func=compute_loss_func,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, args.threshold),
        processing_class=tokenizer,
    )

    trainer.train()

    print("Training completed. Evaluating on test dataset...\n", flush=True)

    test_dataset = StopDecisionDataset(args.test_data_path, tokenizer, args.max_length)
    print(f"Number of test samples: {len(test_dataset)}", flush=True)

    _, _, metrics = trainer.predict(test_dataset)

    print("Test Metrics:", flush=True)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
