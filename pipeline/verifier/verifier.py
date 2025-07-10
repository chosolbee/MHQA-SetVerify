import argparse
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from config import DEBERTA_MAX_LENGTH


class VerifierDataset(Dataset):
    def __init__(self, questions, batch_history, batch_passages, tokenizer, max_length=DEBERTA_MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.question_indices = {}

        for i, (question, history, passages) in enumerate(zip(questions, batch_history, batch_passages)):
            start_idx = len(self.data)
            self._add_question(question, history, passages)
            end_idx = len(self.data)
            self.question_indices[i] = list(range(start_idx, end_idx))

    def _add_question(self, question, history, passages):
        question_text = question["question"]
        prev_context = " [SEP] ".join([h["text"] for h in history]) + " [SEP] " if history else ""
        for passage in passages:
            passage_text = passage["text"]
            prefix = f"Question: {question_text} [SEP] {prev_context}"
            text = prefix + passage_text
            self.data.append({"text": text})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        return encoding


class Verifier:
    def __init__(self, model_id, checkpoint_path, batch_size, max_length=DEBERTA_MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=1,
            problem_type="regression",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.max_length = max_length

        self.training_args = TrainingArguments(
            per_device_eval_batch_size=batch_size,
            remove_unused_columns=False,
            output_dir="results/tmp",  # required but not used for prediction
            fp16=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            data_collator=self.collate_fn,
        )

    def collate_fn(self, batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_verify(self, questions, batch_history, batch_passages):
        test_dataset = VerifierDataset(questions, batch_history, batch_passages, self.tokenizer, self.max_length)

        predictions_output = self.trainer.predict(test_dataset)
        all_logits = predictions_output.predictions.flatten()
        all_preds = torch.sigmoid(torch.tensor(all_logits)).numpy()

        batch_preds = []
        for i in range(len(questions)):
            indices = test_dataset.question_indices.get(i, [])
            q_preds = all_preds[indices]
            batch_preds.append(q_preds)

        return batch_preds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True, help="Checkpoint path for trained model")
    args = parser.parse_args()
    return args


def test(args: argparse.Namespace):
    verifier = Verifier(
        model_id="microsoft/DeBERTa-v3-large",
        checkpoint_path=args.checkpoint_path,
        batch_size=8,
        max_length=DEBERTA_MAX_LENGTH,
    )

    questions = [{"question_id": "2hop__125960_675104", "question": "Where did the Baldevins bryllup director die?"}]
    batch_history = [
        [
            {
                "id": "2hop__125960_675104-00",
                "title": "Baldevins bryllup",
                "text": "Baldevins bryllup () is a 1926 Norwegian comedy film directed by George Schnéevoigt, starring Einar Sissener and Victor Bernau. The film is based on a play by Vilhelm Krag, and tells the story of how Simen Sørensen (Bernau) manages to get his friend Baldevin Jonassen (Sissener) married to the lady next door. The film was renovated in 2006, for the 100-years anniversary of Kristiansand Cinema.",
            }
        ]
    ]
    batch_passages = [
        [
            {
                "text": "Schnéevoigt was born in Copenhagen, Denmark to actress Siri Schnéevoigt, and he is the father of actor and director Alf Schnéevoigt."
            },
            {
                "text": "Le Juge Fayard dit Le Shériff is a 1977 French crime film written and directed by Yves Boisset. The film was inspired by the death of François Renaud."
            },
            {
                "text": "Death Valley is a desert valley located in Eastern California, in the northern Mojave Desert bordering the Great Basin Desert. It is one of the hottest places in the world along with deserts in the Middle East."
            },
            {
                "text": "A Fistful of Death ( ) is a 1971 Italian Western film directed by Demofilo Fidani and starring Klaus Kinski."
            },
        ]
    ]

    batch_preds = verifier.batch_verify(questions, batch_history, batch_passages)

    for question, history, passages, preds in zip(questions, batch_history, batch_passages, batch_preds):
        print(f"1. Question: {question['question']}")
        print("2. History:")
        for doc in history:
            print(f"  Passage: {doc['text']}")
        print("3. Retrieved passages and scores:")
        for doc, pred in zip(passages, preds):
            print(f"  Score: {pred:.2f} | Passage: {doc['text']}")
        print("\n")


if __name__ == "__main__":
    args = parse_args()
    test(args)
