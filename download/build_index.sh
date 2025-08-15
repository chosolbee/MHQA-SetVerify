#!/bin/sh
set -eu

RETRIEVER="${1:-}"

if [ "$RETRIEVER" = "contriever" ]; then
    for ds in hotpotqa 2wikimultihopqa musique; do
        uv run python3 pipeline/contriever/passage_embedder.py \
            --passages data/corpus/$ds/passages/corpus.jsonl \
            --output_dir data/corpus/$ds/contriever_embeddings

        uv run python3 -m pipeline.contriever.passage_retriever \
            --passages data/corpus/$ds/passages/corpus.jsonl \
            --embeddings data/corpus/$ds/contriever_embeddings \
            --save_or_load_index
    done
elif [ "$RETRIEVER" = "bm25" ]; then
    for ds in hotpotqa 2wikimultihopqa musique; do
        uv run python3 -m pipeline.bm25.bm25_retriever \
            --passages data/corpus/$ds/passages/corpus.jsonl \
            --index_path_dir data/corpus/$ds/bm25_index \
            --save_or_load_index
    done
else
    echo "Invalid retriever. Use 'contriever' or 'bm25'." >&2
    exit 1
fi

echo "Indexing completed for retriever: $RETRIEVER"
