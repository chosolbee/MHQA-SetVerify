import os

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
assert WANDB_API_KEY is not None, "Please set the WANDB_API_KEY environment variable."

DEBERTA_MAX_LENGTH = 768

DATASET_PATHS = {
    "hotpotqa": {
        "train": "data/raw/hotpotqa/hotpot_train_v1.1.json",
        "dev": "data/raw/hotpotqa/hotpot_dev_fullwiki_v1.json",
        "test": None,
    },
    "2wikimultihopqa": {
        "train": "data/raw/2wikimultihopqa/train.json",
        "dev": "data/raw/2wikimultihopqa/dev.json",
        "test": "data/raw/2wikimultihopqa/test.json",
    },
    "musique": {
        "train": "data/raw/musique/musique_ans_v1.0_train.jsonl",
        "dev": "data/raw/musique/musique_ans_v1.0_dev.jsonl",
        "test": "data/raw/musique/musique_ans_v1.0_test.jsonl",
    },
}

DATASET_FIELDS = {
    "hotpotqa": {
        "id": "_id",
        "question": "question",
        "answer": "answer",
        "answer_aliases": None,
        "supporting_facts": "supporting_facts",
    },
    "2wikimultihopqa": {
        "id": "_id",
        "question": "question",
        "answer": "answer",
        "answer_aliases": None,
        "supporting_facts": "supporting_facts",
    },
    "musique": {
        "id": "id",
        "question": "question",
        "answer": "answer",
        "answer_aliases": "answer_aliases",
        "supporting_facts": "question_decomposition",
    },
}
