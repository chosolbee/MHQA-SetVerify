import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DEBERTA_MAX_LENGTH


class BoNVerifier:
    def __init__(self, model_id, checkpoint_path, batch_size=8, max_length=DEBERTA_MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=1,
            problem_type="single_label_classification",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.batch_size = batch_size
        self.max_length = max_length

    def batch_verify(self, batch_traces, num_generations=4):
        scores = torch.zeros(len(batch_traces), dtype=torch.float16)
        for i in range(0, len(batch_traces), self.batch_size):
            inputs = self.tokenizer(
                batch_traces[i:i + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            outputs = self.model(**inputs)
            scores[i:i + self.batch_size] = outputs.logits.squeeze(-1).cpu()
        grouped_scores = scores.view(-1, num_generations)
        local_max_indices = torch.argmax(grouped_scores, dim=1)
        global_max_indices = [group_idx * num_generations + local_idx.item()
                              for group_idx, local_idx in enumerate(local_max_indices)]
        return global_max_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True, help="Checkpoint path for trained model")
    args = parser.parse_args()
    return args


def test(args: argparse.Namespace):
    verifier = BoNVerifier(
        model_id="microsoft/DeBERTa-v3-large",
        checkpoint_path=args.checkpoint_path,
        batch_size=4,
        max_length=100,
    )

    batch_traces = ["example 1", "example 2", "example 3", "example 4"]
    num_generations = 2

    indices = verifier.batch_verify(batch_traces, num_generations)
    print(indices)


if __name__ == "__main__":
    args = parse_args()
    test(args)
