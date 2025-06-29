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
    binary_auroc,
    binary_average_precision
)
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
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
from .modules import RankNetLoss, ListNetLoss, LambdaRankLoss, ListMLELoss

class VerifierDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        question_ids = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                question = entry["question"]
                question_id = entry["question_id"]
                passages = entry["passages"]
                passages_text = " [SEP] ".join([p["paragraph_text"] for p in passages])
                text = f"Question: {question} [SEP] {passages_text}"
                
                # Binary classification: coverage == 1.0 -> correct (1), else -> wrong (0)
                coverage = entry.get("coverage", 0.0)
                binary_label = 1 if coverage == 1.0 else 0
                
                # Keep original coverage for ranking losses
                ranking_score = coverage
                
                self.data.append({
                    "text": text, 
                    "binary_label": binary_label,
                    "ranking_score": ranking_score,
                    "question_id": question_id
                })
                question_ids.append(question_id)

        unique_question_ids = sorted(set(question_ids))
        question_id_to_idx = {question_id: idx for idx, question_id in enumerate(unique_question_ids)}
        for item in self.data:
            item["question_id"] = question_id_to_idx[item["question_id"]]

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
            "binary_labels": torch.tensor(item["binary_label"], dtype=torch.float),
            "ranking_scores": torch.tensor(item["ranking_score"], dtype=torch.float),
            "question_id": torch.tensor(item["question_id"], dtype=torch.long),
        }
        return encoding

class GroupSampler(Sampler):
    def __init__(self, dataset, batch_size):
        # self.batch_size = batch_size

        groups = defaultdict(list)
        for idx, item in enumerate(dataset):
            question_id = item["labels"]["question_id"]
            if torch.is_tensor(question_id):
                question_id = question_id.item()
            groups[question_id].append(idx)

        self.groups = list(groups.values())

    def __iter__(self):
        yield from self.groups

    def __len__(self):
        return len(self.groups)


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    binary_labels = torch.stack([x["labels"]["binary_labels"] for x in batch])
    ranking_scores = torch.stack([x["labels"]["ranking_scores"] for x in batch])
    question_ids = torch.stack([x["labels"]["question_id"] for x in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": {
            "binary_labels": binary_labels, 
            "ranking_scores": ranking_scores,
            "question_ids": question_ids
        },
    }


class Metrics:
    def __init__(self):
        self.mrr_metric = RetrievalMRR()
        self.ndcg_metric = RetrievalNormalizedDCG()

    def __call__(self, eval_preds: EvalPrediction):
        logits, labels_dict = eval_preds
        logits = torch.tensor(logits.flatten())
        binary_labels = torch.tensor(labels_dict["binary_labels"].flatten())
        ranking_scores = torch.tensor(labels_dict["ranking_scores"].flatten())
        indexes = torch.tensor(labels_dict["question_ids"].flatten())

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()

        accuracy = binary_accuracy(predictions, binary_labels).item()
        precision = binary_precision(predictions, binary_labels).item()
        recall = binary_recall(predictions, binary_labels).item()
        f1 = binary_f1_score(predictions, binary_labels).item()
        
        try:
            auroc = binary_auroc(probabilities, binary_labels.long()).item()
        except:
            auroc = 0.0
            
        try:
            avg_precision = binary_average_precision(probabilities, binary_labels.long()).item()
        except:
            avg_precision = 0.0

        # Group-wise metrics
        group_accuracies = []
        binary_correct = binary_total = 0
        ranking_correct = ranking_total = 0

        for idx in torch.unique(indexes):
            mask = indexes == idx
            p_probs = probabilities[mask]
            p_preds = predictions[mask]
            b_labels = binary_labels[mask]
            r_scores = ranking_scores[mask]

            if len(p_probs) < 2:
                continue

            # Group accuracy (binary classification)
            group_acc = (p_preds == b_labels).float().mean().item()
            group_accuracies.append(group_acc)

            # Binary pairwise ranking accuracy
            pred_diff = p_probs.unsqueeze(0) - p_probs.unsqueeze(1)
            binary_diff = b_labels.unsqueeze(0) - b_labels.unsqueeze(1)
            pred_sign = torch.sign(pred_diff)
            binary_sign = torch.sign(binary_diff)
            valid_binary_mask = binary_sign != 0
            binary_correct += ((pred_sign == binary_sign) & valid_binary_mask).sum().item()
            binary_total += valid_binary_mask.sum().item()

            # Ranking pairwise accuracy (using original coverage scores)
            ranking_diff = r_scores.unsqueeze(0) - r_scores.unsqueeze(1)
            ranking_sign = torch.sign(ranking_diff)
            valid_ranking_mask = ranking_sign != 0
            ranking_correct += ((pred_sign == ranking_sign) & valid_ranking_mask).sum().item()
            ranking_total += valid_ranking_mask.sum().item()

        # Retrieval metrics (using ranking scores as targets)
        try:
            mrr = self.mrr_metric(probabilities, ranking_scores, indexes).item()
        except:
            mrr = 0.0
            
        try:
            ndcg = self.ndcg_metric(probabilities, ranking_scores, indexes).item()
        except:
            ndcg = 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
            "avg_precision": avg_precision,
            "group_accuracy": np.mean(group_accuracies) if group_accuracies else 0.0,
            "binary_pairwise_accuracy": binary_correct / binary_total if binary_total > 0 else 0,
            "ranking_pairwise_accuracy": ranking_correct / ranking_total if ranking_total > 0 else 0,
            "mrr": mrr,
            "ndcg": ndcg,
        }


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        losses_arg = kwargs.pop("losses", "bce")
        if isinstance(losses_arg, list):
            self.losses = losses_arg
        else:
            self.losses = [l.strip() for l in losses_arg.split(",")]

        loss_weights_arg = kwargs.pop("loss_weights", [1.0])
        self.loss_weights = {loss: weight for loss, weight in zip(self.losses, loss_weights_arg)}

        margin = kwargs.pop("margin", 0.1)
        sigma = kwargs.pop("sigma", 1.0)

        self.loss_fns = {}
        for loss in self.losses:
            if loss == "bce":
                self.loss_fns["bce"] = nn.BCEWithLogitsLoss()
            elif loss == "focal":
                alpha = kwargs.pop("focal_alpha", 1.0)
                gamma = kwargs.pop("focal_gamma", 2.0)
                self.loss_fns["focal"] = self._focal_loss_fn(alpha, gamma)
            elif loss == "margin":
                self.loss_fns["margin"] = nn.MarginRankingLoss(margin=margin)
            elif loss == "ranknet":
                self.loss_fns["ranknet"] = RankNetLoss(sigma=sigma)
            elif loss == "listnet":
                self.loss_fns["listnet"] = ListNetLoss()
            elif loss == "lambdarank":
                self.loss_fns["lambdarank"] = LambdaRankLoss(sigma=sigma)
            elif loss == "listmle":
                self.loss_fns["listmle"] = ListMLELoss()
            else:
                raise ValueError(f"Unknown loss function: {loss}")

        kwargs.pop("focal_alpha", None)
        kwargs.pop("focal_gamma", None)

        super().__init__(*args, **kwargs)

    def _focal_loss_fn(self, alpha, gamma):
        def focal_loss(logits, targets):
            bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = alpha * (1 - pt) ** gamma * bce_loss
            return focal_loss.mean()
        return focal_loss

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: Provided train_dataset is None")

        return DataLoader(
            self.train_dataset,
            batch_sampler=GroupSampler(self.train_dataset, self.args.per_device_train_batch_size),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_dict = inputs.pop("labels", None)

        outputs = model(**inputs)

        logits = outputs.logits.squeeze(-1)
        probabilities = torch.sigmoid(logits)
        
        binary_labels = labels_dict["binary_labels"]
        ranking_scores = labels_dict["ranking_scores"]
        indexes = labels_dict["question_ids"]

        total_loss = 0.0
        component_losses = {}

        for key, loss_fn in self.loss_fns.items():
            if key in ('bce', 'focal'):
                loss_value = loss_fn(logits, binary_labels)
            elif key == "margin":
                labels_diff = ranking_scores.unsqueeze(0) - ranking_scores.unsqueeze(1)
                mask = labels_diff > 0 
                if mask.any():
                    i_idx, j_idx = mask.nonzero(as_tuple=True)
                    prob_i = probabilities[i_idx]
                    prob_j = probabilities[j_idx]
                    target = torch.ones_like(prob_i)
                    loss_value = loss_fn(prob_i, prob_j, target)
                else:
                    loss_value = torch.tensor(0.0, device=logits.device)
            elif key in ("ranknet", "listnet", "lambdarank", "listmle"):
                # Convert binary labels to continuous scores for ranking losses
                loss_value = loss_fn(probabilities, ranking_scores, indexes)
            else:
                raise ValueError(f"Unknown loss function: {key}")

            weighted_loss = self.loss_weights[key] * loss_value
            total_loss += weighted_loss
            component_losses[key] = loss_value.item()

        if return_outputs:
            outputs.loss = total_loss
            return total_loss, outputs
        return total_loss


class PrintCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"\n[Step {state.global_step}]")
            for key, value in logs.items():
                print(f"  {key}: {value:.4f}")
            print("-" * 40)

def validate_losses(losses_str):
    allowed_losses = {'bce', 'focal', 'margin', 'ranknet', 'listnet', 'lambdarank', 'listmle'}
    losses = [loss.strip() for loss in losses_str.split(",")]
    for loss in losses:
        if loss not in allowed_losses:
            raise argparse.ArgumentTypeError(f"Invalid loss function: {loss}. Allowed values: {allowed_losses}")
    return losses

def validate_loss_weights(weights_input):
    if isinstance(weights_input, list):
        return [float(w) for w in weights_input]
    else:
        return [float(w.strip()) for w in weights_input.split(",")]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Verifier Binary Classification Train")

    parser.add_argument("--train-data-path", help="Training Dataset Path", type=str, required=True)
    parser.add_argument("--eval-data-path", help="Evaluation Dataset Path", type=str, required=True)
    parser.add_argument("--test-data-path", help="Test Dataset Path", type=str, required=True)
    parser.add_argument("--trainer-output-dir", help="Training Output Path", type=str)
    parser.add_argument("--max-length", help="Max Length of Tokenizer", type=int, default=DEBERTA_MAX_LENGTH)
    parser.add_argument("--learning-rate", help="Learning Rate", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler-type", help="Learning Rate Scheduler Type", type=str, default="cosine")
    parser.add_argument("--warmup-ratio", help="Warmup Ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", help="Weight Decay", type=float, default=0.01)
    parser.add_argument("--batch-size", help="Batch Size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", help="Gradient Accumulation Steps", type=int, default=4)
    parser.add_argument("--num-epochs", help="Number of Epochs", type=int, default=3)
    parser.add_argument("--fp16", help="Use FP16", action="store_true")
    parser.add_argument("--losses", help="Comma-separated list of losses", type=validate_losses, default="bce")
    parser.add_argument("--loss-weights", help="Comma-separated list of loss weights corresponding to the losses", type=validate_loss_weights, default="1.0")
    parser.add_argument("--margin", help="Margin for Ranking Loss", type=float, default=0.1)
    parser.add_argument("--sigma", help="Sigma for RankNet or LambdaRank", type=float, default=1.0)
    parser.add_argument("--focal-alpha", help="Alpha for Focal Loss", type=float, default=1.0)
    parser.add_argument("--focal-gamma", help="Gamma for Focal Loss", type=float, default=2.0)
    parser.add_argument("--run-name", help="Custom WandB run name", type=str, default=None)

    args = parser.parse_args()

    if len(args.losses) != len(args.loss_weights):
        parser.error("Number of loss weights must match number of losses")

    return args


def main():
    args = parse_arguments()

    print("Selected Loss Functions and Weights:")
    for loss, weight in zip(args.losses, args.loss_weights): 
        print(f"  {loss}: {weight}")

    model_name = "microsoft/deberta-v3-large"

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1 or local_rank == 0: 
        wandb.init(
            project="verifier-binary-classification",
            entity=WANDB_ENTITY,
            name=args.run_name,
            config={
                "model_name": model_name,
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
                "lr_scheduler_type": args.lr_scheduler_type,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "epochs": args.num_epochs,
                "fp16": args.fp16,
                "losses": args.losses,       
                "loss_weights": args.loss_weights,
                "margin": args.margin,
                "sigma": args.sigma,
                "focal_alpha": getattr(args, 'focal_alpha', 1.0),
                "focal_gamma": getattr(args, 'focal_gamma', 2.0),
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
        report_to=["wandb"],
        run_name=args.run_name,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True, 
        ddp_find_unused_parameters=False,
        fp16=args.fp16,
    )

    train_dataset = VerifierDataset(args.train_data_path, tokenizer, args.max_length)
    eval_dataset = VerifierDataset(args.eval_data_path, tokenizer, args.max_length)

    print_callback = PrintCallback()
    compute_metrics = Metrics()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[print_callback],
        losses=args.losses,       
        loss_weights=args.loss_weights,
        margin=args.margin,
        sigma=args.sigma,
        focal_alpha=getattr(args, 'focal_alpha', 1.0),
        focal_gamma=getattr(args, 'focal_gamma', 2.0),
    )

    trainer.train()

    print("Training completed. Evaluating on test dataset...\n")

    test_dataset = VerifierDataset(args.test_data_path, tokenizer, args.max_length)
    _, _, metrics = trainer.predict(test_dataset)

    print("Test Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()