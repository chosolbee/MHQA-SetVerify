import os
import json
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_f1_score,
    binary_auroc
)
from torchmetrics.functional.regression import (
    spearman_corrcoef,
    kendall_rank_corrcoef
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    TrainerCallback,
)
import wandb
from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH

class BinaryVerifierDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=768, f1_threshold=0.7):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.f1_threshold = f1_threshold
        self.data = []
        question_ids = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                question = entry["question"]
                question_id = entry.get("question_id", f"q_{len(self.data)}")
                trace = entry["trace"]
                f1_score = entry["f1"]
                
                text = f"Question: {question} [SEP] Trace: {trace}"
                
                label = 1 if f1_score > f1_threshold else 0
                
                self.data.append({
                    "text": text, 
                    "label": label, 
                    "question_id": question_id,
                    "f1_score": f1_score
                })
                question_ids.append(question_id)

        unique_question_ids = sorted(set(question_ids))
        question_id_to_idx = {question_id: idx for idx, question_id in enumerate(unique_question_ids)}
        for item in self.data:
            item["question_id_idx"] = question_id_to_idx[item["question_id"]]

        print(f"Loaded {len(self.data)} examples")
        print(f"Positive examples (f1 > {f1_threshold}): {sum(item['label'] for item in self.data)}")
        print(f"Negative examples (f1 <= {f1_threshold}): {sum(1 - item['label'] for item in self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = {
            "labels": torch.tensor(item["label"], dtype=torch.float),
            "question_id": torch.tensor(item["question_id_idx"], dtype=torch.long),
            "f1_score": torch.tensor(item["f1_score"], dtype=torch.float),
        }
        return encoding

class QuestionGroupSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        groups = defaultdict(list)
        for idx, item in enumerate(dataset):
            question_id = item["labels"]["question_id"]
            if torch.is_tensor(question_id):
                question_id = question_id.item()
            groups[question_id].append(idx)

        self.groups = list(groups.values())
        
        self.batches = []
        current_batch = []
        
        for group in self.groups:
            if len(current_batch) + len(group) <= batch_size:
                current_batch.extend(group)
            else:
                if current_batch:
                    self.batches.append(current_batch)
                current_batch = group[:batch_size]
                
            if len(current_batch) >= batch_size:
                self.batches.append(current_batch[:batch_size])
                current_batch = current_batch[batch_size:]
        
        if current_batch:
            self.batches.append(current_batch)

    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.batches)
        
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"]["labels"] for x in batch])
    question_ids = torch.stack([x["labels"]["question_id"] for x in batch])
    f1_scores = torch.stack([x["labels"]["f1_score"] for x in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": {
            "labels": labels, 
            "question_ids": question_ids,
            "f1_scores": f1_scores
        },
    }

class BinaryMetrics:
    def __init__(self):
        pass

    def __call__(self, eval_preds: EvalPrediction):
        logits, labels_dict = eval_preds
        logits = torch.tensor(logits.flatten())
        labels = torch.tensor(labels_dict["labels"].flatten())
        indexes = torch.tensor(labels_dict["question_ids"].flatten())
        f1_scores = torch.tensor(labels_dict["f1_scores"].flatten())

        predictions = torch.sigmoid(logits)
        binary_preds = (predictions > 0.5).float()

        metrics = {
            "accuracy": binary_accuracy(binary_preds, labels).item(),
            "precision": binary_precision(binary_preds, labels).item(),
            "recall": binary_recall(binary_preds, labels).item(),
            "f1": binary_f1_score(binary_preds, labels).item(),
            "auroc": binary_auroc(predictions, labels.long()).item(),
        }

        spearman_scores = []
        kendall_scores = []
        correct_rankings = 0
        total_rankings = 0

        for idx in torch.unique(indexes):
            mask = indexes == idx
            p = predictions[mask]
            l = labels[mask]
            f1_vals = f1_scores[mask]

            if len(p) < 2:
                continue

            if len(torch.unique(l)) > 1:
                spearman_scores.append(spearman_corrcoef(p, l).item())
                kendall_scores.append(kendall_rank_corrcoef(p, l).item())

            spearman_f1 = spearman_corrcoef(p, f1_vals)
            if not torch.isnan(spearman_f1):
                best_pred_idx = torch.argmax(p)
                best_f1_idx = torch.argmax(f1_vals)
                if best_pred_idx == best_f1_idx:
                    correct_rankings += 1
                total_rankings += 1

        if spearman_scores:
            metrics["spearman"] = np.nanmean(spearman_scores)
            metrics["kendall"] = np.nanmean(kendall_scores)
        else:
            metrics["spearman"] = 0.0
            metrics["kendall"] = 0.0

        metrics["ranking_accuracy"] = correct_rankings / total_rankings if total_rankings > 0 else 0.0

        return metrics

class BinaryVerifierTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        loss_type = kwargs.pop("loss_type", "bce")
        pos_weight = kwargs.pop("pos_weight", None)
        focal_alpha = kwargs.pop("focal_alpha", 0.25)
        focal_gamma = kwargs.pop("focal_gamma", 2.0)
        
        if loss_type == "bce":
            if pos_weight is not None:
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: Provided train_dataset is None")

        sampler = QuestionGroupSampler(
            self.train_dataset, 
            self.args.per_device_train_batch_size,
            shuffle=True
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_dict = inputs.pop("labels", None)
        outputs = model(**inputs)
        
        logits = outputs.logits.squeeze(-1)
        labels = labels_dict["labels"]
        
        loss = self.loss_fn(logits, labels)
        
        if return_outputs:
            outputs.loss = loss
            return loss, outputs
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class PrintCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"\n[Step {state.global_step}]")
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print("-" * 40)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Binary Verifier Training")

    parser.add_argument("--train-data-path", help="Training Dataset Path", type=str, required=True)
    parser.add_argument("--eval-data-path", help="Evaluation Dataset Path", type=str, required=True)
    parser.add_argument("--test-data-path", help="Test Dataset Path", type=str, required=True)
    parser.add_argument("--trainer-output-dir", help="Training Output Path", type=str, required=True)
    parser.add_argument("--max-length", help="Max Length of Tokenizer", type=int, default=DEBERTA_MAX_LENGTH)
    parser.add_argument("--f1-threshold", help="F1 threshold for binary classification", type=float, default=0.7)
    parser.add_argument("--learning-rate", help="Learning Rate", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler-type", help="Learning Rate Scheduler Type", type=str, default="cosine")
    parser.add_argument("--warmup-ratio", help="Warmup Ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", help="Weight Decay", type=float, default=0.01)
    parser.add_argument("--batch-size", help="Batch Size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", help="Gradient Accumulation Steps", type=int, default=4)
    parser.add_argument("--num-epochs", help="Number of Epochs", type=int, default=3)
    parser.add_argument("--fp16", help="Use FP16", action="store_true")
    parser.add_argument("--loss-type", help="Loss function type", choices=["bce", "focal"], default="bce")
    parser.add_argument("--pos-weight", help="Positive class weight for BCE loss", type=float, default=None)
    parser.add_argument("--focal-alpha", help="Alpha parameter for focal loss", type=float, default=0.25)
    parser.add_argument("--focal-gamma", help="Gamma parameter for focal loss", type=float, default=2.0)
    parser.add_argument("--run-name", help="Custom WandB run name", type=str, default=None)

    return parser.parse_args()

def main():
    args = parse_arguments()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

    print(f"Binary Classification Setup:")
    print(f"  F1 Threshold: {args.f1_threshold}")
    print(f"  Loss Type: {args.loss_type}")
    if args.pos_weight:
        print(f"  Positive Weight: {args.pos_weight}")

    model_name = "microsoft/deberta-v3-large"

    if local_rank == -1 or local_rank == 0:
        wandb.init(
            project="binary-verifier-train",
            entity=WANDB_ENTITY,
            name=args.run_name,
            config={
                "model_name": model_name,
                "max_length": args.max_length,
                "f1_threshold": args.f1_threshold,
                "learning_rate": args.learning_rate,
                "lr_scheduler_type": args.lr_scheduler_type,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "epochs": args.num_epochs,
                "fp16": args.fp16,
                "loss_type": args.loss_type,
                "pos_weight": args.pos_weight,
            },
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1,
        problem_type="single_label_classification"
    )

    if local_rank != -1:
        model = model.to(f'cuda:{local_rank}')

    training_args = TrainingArguments(
        output_dir=args.trainer_output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_strategy="steps",
        logging_steps=100,
        report_to=["wandb"] if (local_rank == -1 or local_rank == 0) else [],
        run_name=args.run_name,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        ddp_find_unused_parameters=False,
        fp16=args.fp16,
        local_rank=local_rank,
        dataloader_pin_memory=False, 
    )

    train_dataset = BinaryVerifierDataset(
        args.train_data_path, tokenizer, args.max_length, args.f1_threshold
    )
    eval_dataset = BinaryVerifierDataset(
        args.eval_data_path, tokenizer, args.max_length, args.f1_threshold
    )

    if args.pos_weight is None and args.loss_type == "bce":
        pos_count = sum(item["label"] for item in train_dataset.data)
        neg_count = len(train_dataset.data) - pos_count
        if pos_count > 0:
            args.pos_weight = neg_count / pos_count
            if local_rank == -1 or local_rank == 0:
                print(f"Computed positive class weight: {args.pos_weight:.3f}")

    print_callback = PrintCallback()
    compute_metrics = BinaryMetrics()

    trainer = BinaryVerifierTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[print_callback],
        loss_type=args.loss_type,
        pos_weight=args.pos_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    print("Starting training...")
    trainer.train()

    if local_rank == -1 or local_rank == 0:
        print("Training completed. Evaluating on test dataset...\n")

        test_dataset = BinaryVerifierDataset(
            args.test_data_path, tokenizer, args.max_length, args.f1_threshold
        )
        _, _, test_metrics = trainer.predict(test_dataset)

        print("Test Metrics:")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}")

    trainer.save_model()
    tokenizer.save_pretrained(args.trainer_output_dir)

    if local_rank == -1 or local_rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()