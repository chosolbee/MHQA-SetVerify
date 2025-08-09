import pickle
import os
import sys
import re
import argparse
from typing import List, Union, Dict, Any
from rank_bm25 import BM25Okapi
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from contriever.utils.utils import load_passages


class BM25Retriever(object):
    def __init__(
        self,
        passage_path: str,
        passage_embedding_path: str = None,
        index_path_dir: str = None,
        model_type: str = None,
        model_path: str = None,
        save_or_load_index: bool = True,
        batch_size: int = 128,
        embed_vector_dim: int = None,
        index_type: str = "Flat",
        max_search_batch_size: int = 2048,
        k1: float = 1.5,
        b: float = 0.8,
        epsilon: float = 0.2
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        if index_path_dir is None:
            index_path_dir = os.path.join(os.path.dirname(os.path.dirname(passage_path)), "bm25_index")

        self.index_path = os.path.join(index_path_dir, "bm25_index.pkl")

        if save_or_load_index and self._index_exists():
            print(f"Loading BM25 index from {self.index_path}")
            self._load_index()
        else:
            print(f"Building BM25 index from {passage_path}")
            self._build_index(passage_path)
            if save_or_load_index:
                print(f"Saving BM25 index to {self.index_path}")
                self._save_index()

        print(f"Loading passages from {passage_path}")
        passages = load_passages(passage_path)
        self.passage_map = {p["id"]: p for p in passages}
        print(f"Loaded {len(passages)} passages.")

    def _index_exists(self) -> bool:
        return os.path.exists(self.index_path)

    def _preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        return tokens

    def _build_index(self, passage_path: str):
        passages = load_passages(passage_path)

        tokenized_docs = []
        for passage in passages:
            full_text = f"{passage.get('title', '')} {passage['text']}"
            tokens = self._preprocess_text(full_text)
            tokenized_docs.append(tokens)

        self.bm25 = BM25Okapi(
            tokenized_docs,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )
        self.tokenized_docs = tokenized_docs
        self.passages = passages

    def _save_index(self):
        index_data = {
            'bm25': self.bm25,
            'tokenized_docs': self.tokenized_docs,
            'passages': self.passages,
            'k1': self.k1,
            'b': self.b,
            'epsilon': self.epsilon
        }
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(index_data, f)

    def _load_index(self):
        with open(self.index_path, 'rb') as f:
            index_data = pickle.load(f)

        self.bm25 = index_data['bm25']
        self.tokenized_docs = index_data['tokenized_docs']
        self.passages = index_data['passages']
        self.k1 = index_data.get('k1', self.k1)
        self.b = index_data.get('b', self.b)
        self.epsilon = index_data.get('epsilon', self.epsilon)

    def search(self, query: Union[str, List[str]], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        queries = [query] if isinstance(query, str) else query

        results = []
        for q in queries:
            tokenized_query = self._preprocess_text(q)
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = scores.argsort()[-top_k:][::-1]

            query_results = []
            for idx in top_indices:
                if idx < len(self.passages):
                    passage_id = self.passages[idx]["id"]
                    if passage_id in self.passage_map:
                        doc = self.passage_map[passage_id].copy()
                        doc['score'] = float(scores[idx])
                        query_results.append(doc)

            results.append(query_results[:top_k])
        return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--passages", type=str, required=True, help="document file path"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bm25",
        help="Retrieval model type (bm25)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Not used for BM25 but kept for compatibility",
    )
    parser.add_argument(
        "--index_path_dir",
        type=str,
        default=None,
        help="Directory to save or load the BM25 index",
    )
    parser.add_argument("--save_or_load_index", action="store_true")
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Not used for BM25 but kept for compatibility"
    )
    parser.add_argument("--query", type=str, help="query")
    parser.add_argument(
        "--k1", type=float, default=1.5, help="BM25 k1 parameter"
    )
    parser.add_argument(
        "--b", type=float, default=0.8, help="BM25 b parameter" 
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.2, help="BM25 epsilon parameter"
    )

    args = parser.parse_args()
    return args


def test(opt):
    retriever = BM25Retriever(
        passage_path=opt.passages,
        index_path_dir=opt.index_path_dir,
        save_or_load_index=opt.save_or_load_index,
        k1=opt.k1,
        b=opt.b,
        epsilon=opt.epsilon
    )

    if opt.query is None:
        queries = [
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            "What is the difference between llama and alpaca?",
        ]
    else:
        queries = [opt.query]

    docs = retriever.search(queries, 20)
    print(docs)


if __name__ == "__main__":
    options = parse_args()
    test(options)
