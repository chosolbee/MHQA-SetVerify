import argparse
import os
import sys
from typing import List, Union
import bm25s
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
        k1: float = 1.2,
        b: float = 0.75,
        method: str = "lucene",
        use_mmap: bool = False,
    ):
        self.k1 = k1
        self.b = b
        self.method = method
        self.use_mmap = use_mmap

        if index_path_dir is None:
            index_path_dir = os.path.join(os.path.dirname(os.path.dirname(passage_path)), "bm25_index")
        self.index_path = os.path.join(index_path_dir, "bm25s_index")

        self._load_passages_and_passage_map(passage_path)

        if save_or_load_index and self._index_exists():
            print(f"Loading index from {self.index_path}")
            self._load_index()
        else:
            print(f"Building index from {passage_path}")
            self._build_index()
            if save_or_load_index:
                print(f"Saving index to {self.index_path}")
                self._save_index()

    def _index_exists(self):
        return os.path.exists(self.index_path) and os.path.isdir(self.index_path)

    def _load_passages_and_passage_map(self, passage_path: str):
        self.passages = load_passages(passage_path)
        self.passage_map = {p["id"]: p for p in self.passages}
        print(f"Loaded {len(self.passages)} passages from {passage_path}.")

    def _build_index(self):
        corpus = []
        for passage in self.passages:
            title = passage.get('title', '').strip()
            text = passage.get('text', '').strip()
            full_text = f"{title} {text}".strip() if title else text
            corpus.append(full_text)

        print("Tokenizing corpus...")
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", return_ids=False)

        print("Creating BM25S model and indexing...")
        self.bm25 = bm25s.BM25(k1=self.k1, b=self.b, method=self.method)
        self.bm25.index(corpus_tokens)

        print(f"BM25S indexing completed. Method: {self.method}")

    def _save_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.bm25.save(self.index_path)

    def _load_index(self):
        self.bm25 = bm25s.BM25.load(
            self.index_path, 
            load_corpus=False,
            mmap=self.use_mmap
        )

        print(f"BM25S index loaded. Method: {self.bm25.method}, Memory-mapped: {self.use_mmap}")

    def search(self, query: Union[str, List[str]], top_k: int = 10):
        query = [query] if isinstance(query, str) else query
        query_tokens = bm25s.tokenize(query, stopwords="en", return_ids=False)

        docs, scores = self.bm25.retrieve(
            query_tokens,
            corpus=self.passages,
            k=top_k,
            show_progress=False
        )

        results = []
        for row_docs, row_scores in zip(docs, scores):
            doc_list = []
            for doc, score in zip(row_docs, row_scores):
                result_doc = doc.copy()
                result_doc['score'] = float(score)
                doc_list.append(result_doc)
            results.append(doc_list[:top_k])

        return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--passages", type=str, required=True, help="document file path"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bm25s",
        help="Retrieval model type (bm25s)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Not used for BM25S but kept for compatibility",
    )
    parser.add_argument(
        "--index_path_dir",
        type=str,
        default=None,
        help="Directory to save or load the BM25S index",
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
        help="Not used for BM25S but kept for compatibility"
    )
    parser.add_argument("--query", type=str, help="query")
    parser.add_argument(
        "--k1", type=float, default=1.2, help="BM25 k1 parameter"
    )
    parser.add_argument(
        "--b", type=float, default=0.75, help="BM25 b parameter" 
    )
    parser.add_argument(
        "--method", type=str, default="lucene", 
        choices=["lucene", "robertson", "atire", "bm25l", "bm25+"],
        help="BM25 variant method"
    )
    parser.add_argument(
        "--use_mmap", action="store_true", 
        help="Use memory mapping for large indices"
    )
    args = parser.parse_args()
    return args


def test(opt):
    retriever = BM25Retriever(
        opt.passages,
        index_path_dir=opt.index_path_dir,
        save_or_load_index=opt.save_or_load_index,
        k1=opt.k1,
        b=opt.b,
        method=opt.method,
        use_mmap=opt.use_mmap,
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
