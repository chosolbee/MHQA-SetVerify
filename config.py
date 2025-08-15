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
        "eval_subsampled": "data/raw/hotpotqa/hotpot_dev_fullwiki_v1_eval_subsampled.json",
        "test_subsampled": "data/raw/hotpotqa/hotpot_dev_fullwiki_v1_test_subsampled.json",
        "passages": "data/corpus/hotpotqa/passages/corpus.jsonl",
        "contriever_embeddings": "data/corpus/hotpotqa/contriever_embeddings",
        "bm25_index": "data/corpus/hotpotqa/bm25_index",
    },
    "2wikimultihopqa": {
        "train": "data/raw/2wikimultihopqa/train.json",
        "dev": "data/raw/2wikimultihopqa/dev.json",
        "test": "data/raw/2wikimultihopqa/test.json",
        "eval_subsampled": "data/raw/2wikimultihopqa/dev_eval_subsampled.json",
        "test_subsampled": "data/raw/2wikimultihopqa/dev_test_subsampled.json",
        "passages": "data/corpus/2wikimultihopqa/passages/corpus.jsonl",
        "contriever_embeddings": "data/corpus/2wikimultihopqa/contriever_embeddings",
        "bm25_index": "data/corpus/2wikimultihopqa/bm25_index",
    },
    "musique": {
        "train": "data/raw/musique/musique_ans_v1.0_train.jsonl",
        "dev": "data/raw/musique/musique_ans_v1.0_dev.jsonl",
        "test": "data/raw/musique/musique_ans_v1.0_test.jsonl",
        "eval_subsampled": "data/raw/musique/musique_ans_v1.0_dev_eval_subsampled.jsonl",
        "test_subsampled": "data/raw/musique/musique_ans_v1.0_dev_test_subsampled.jsonl",
        "passages": "data/corpus/musique/passages/corpus.jsonl",
        "contriever_embeddings": "data/corpus/musique/contriever_embeddings",
        "bm25_index": "data/corpus/musique/bm25_index",
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
