import os
import json
import math
import argparse
from typing import List, Dict
from collections import defaultdict, Counter

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .adaptive_pipeline import (
    NoRetrieval, SingleRetrieval, MultiRetrieval,
    create_retriever, create_llm_components, load_dataset
)
from config import DEBERTA_MAX_LENGTH, DATASET_PATHS, DATASET_FIELDS

from pipeline.utils import print_metrics

LABEL_TO_STRATEGY = {"A": "no", "B": "single", "C": "multi"}

def classify_questions(model, tokenizer, questions, fields, batch_size=64, device="cuda"):
    model.eval().to(device)
    labels = []
    ids = []
    with torch.no_grad():
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            texts = [q[fields["question"]].strip() for q in batch]
            enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

            out = model.generate(
                **enc,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False
            )
            scores = out.scores[0]

            a_id = tokenizer("A").input_ids[0]
            b_id = tokenizer("B").input_ids[0]
            c_id = tokenizer("C").input_ids[0]

            triple = torch.stack([scores[:, a_id], scores[:, b_id], scores[:, c_id]], dim=0)
            pred = torch.argmax(torch.softmax(triple, dim=0), dim=0)
            
            for j, q in enumerate(batch):
                lab = ["A", "B", "C"][int(pred[j].item())]
                labels.append(lab)
                ids.append(q[fields["id"]])

    return ids, labels

def run_group(strategy, group_questions, args, retriever, reranker, reasoner, qa_reader, fields):

    if not group_questions:

        return {
            "retrieval": {"em":[[],[],[]], "precision":[[],[],[]], "recall":[[],[],[]], "f1":[[],[],[]]},
            "answer": {"em":[[],[],[]], "f1":[[],[],[]], "acc":[[],[],[]]}
        }

    common = dict(
        reasoner=reasoner,
        qa_reader=qa_reader,
        questions=group_questions,
        max_search=args.max_search,
        search_batch_size=args.search_batch_size,
        traces_path=args.traces_path,
        log_trace=args.log_trace,
        fields=fields,
    )

    if strategy == "no":
        return NoRetrieval.run_batch(**common)
    elif strategy == "single":
        common.update(dict(
            retriever=retriever,
            reranker=reranker,
            max_docs=args.max_docs,
            allow_duplicate_docs=args.allow_duplicate_docs,
            stop_log_path=args.stop_log_path
        ))
        return SingleRetrieval.run_batch(**common)
    elif strategy == "multi":
        common.update(dict(
            retriever=retriever,
            reranker=reranker,
            max_docs=args.max_docs,
            allow_duplicate_docs=args.allow_duplicate_docs,
            stop_log_path=args.stop_log_path,
        ))
        return MultiRetrieval.run_batch(max_iterations=args.max_iterations, **common)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def merge_metrics(dst, src):
    for mt in ["retrieval", "answer"]:
        for name, hops in src[mt].items():
            for h in range(3):
                dst[mt][name][h].extend(hops[h])
    return dst

def save_predictions(path: str, ids: List[str], labels: List[str]):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, lab in zip(ids, labels):
            rec = {
                "id": str(qid),
                "label": lab,
                "strategy": LABEL_TO_STRATEGY[lab],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_predictions(path: str) -> Dict[str, str]:
    pred_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj["id"])
            strat = obj.get("strategy")
            if strat is None and "label" in obj:
                strat = LABEL_TO_STRATEGY[obj["label"]]
            pred_map[qid] = strat
    return pred_map

def parse_args():
    p = argparse.ArgumentParser("Adaptive orchestrator with classifier")

    p.add_argument("--classifier_model", type=str, required=True)
    p.add_argument("--classifier_batch_size", type=int, default=64)

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--dataset-type", type=str, required=True, choices=["train","dev","test","eval_subsampled","test_subsampled","adaptive-rag_subsampled"])
    p.add_argument("--dataset-path", type=str)
    p.add_argument("--icl-examples-path", type=str, required=True)

    p.add_argument("--retriever-type", type=str, default="contriever", choices=["contriever","bm25"])
    p.add_argument("--passages", type=str)
    p.add_argument("--contriever-embeddings", type=str)
    p.add_argument("--bm25-index-path-dir", type=str)
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.8)
    p.add_argument("--bm25-method", type=str, default="lucene", choices=["lucene","robertson","atire","bm25l","bm25+"])
    p.add_argument("--bm25-use-mmap", action="store_true")

    p.add_argument("--max-search", type=int, default=10)
    p.add_argument("--search-batch-size", type=int, default=16)
    p.add_argument("--max-docs", type=int, default=1)
    p.add_argument("--allow-duplicate-docs", action="store_true")

    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--traces-path", type=str)
    p.add_argument("--stop-log-path", type=str)
    p.add_argument("--log-trace", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--vllm-model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--vllm-tp-size", type=int, default=1)
    p.add_argument("--vllm-quantization", type=str)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--vllm-max-model-len", type=int)

    p.add_argument("--openai-model-id", type=str, default="gpt-4o-mini-2024-07-18")
    p.add_argument("--openai-max-retries", type=int, default=1)
    p.add_argument("--openai-batch-timeout", type=int, default=600)
    p.add_argument("--openai-total-timeout", type=int, default=60)
    p.add_argument("--openai-connect-timeout", type=int, default=10)
    p.add_argument("--openai-max-keepalive-connections", type=int, default=10)
    p.add_argument("--openai-max-connections", type=int, default=20)
    p.add_argument("--openai-max-concurrent", type=int, default=3)

    p.add_argument("--reasoner-max-gen-length", type=int, default=400)
    p.add_argument("--reasoner-temperature", type=float, default=0.0)
    p.add_argument("--reasoner-top-p", type=float, default=1.0)
    p.add_argument("--reasoner-provider", type=str, default="vllm", choices=["vllm","openai"])

    p.add_argument("--qareader-max-gen-length", type=int, default=400)
    p.add_argument("--qareader-temperature", type=float, default=0.0)
    p.add_argument("--qareader-top-p", type=float, default=1.0)
    p.add_argument("--qareader-provider", type=str, default="vllm", choices=["vllm","openai"])
    p.add_argument("--qareader-disable", action="store_true")

    p.add_argument("--reranker-model-id", type=str, default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--reranker-batch-size", type=int, default=8)
    p.add_argument("--reranker-max-length", type=int, default=4096)
    p.add_argument("--reranker-disable", action="store_true")

    p.add_argument("--save-preds", type=str, required=True, help="save classify result")

    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    questions = load_dataset(args)
    fields = DATASET_FIELDS[args.dataset]
    id_field = fields["id"]

    buckets: Dict[str, List[dict]] = {"no": [], "single": [], "multi": []}


    clf_tok = AutoTokenizer.from_pretrained(args.classifier_model, use_fast=True)
    clf_mod = AutoModelForSeq2SeqLM.from_pretrained(args.classifier_model)

    ids, labels = classify_questions(
        clf_mod, clf_tok, questions, fields,
        batch_size=args.classifier_batch_size, device=device
    )

    cnt = Counter(labels)
    total = len(labels)
    a, b, c = cnt.get("A", 0), cnt.get("B", 0), cnt.get("C", 0)
    print(f"[CLASSIFY] total={total} | A(no)={a} | B(single)={b} | C(multi)={c}")
    if total > 0:
        print(f"[CLASSIFY %] A={a/total:.1%} | B={b/total:.1%} | C={c/total:.1%}")


    save_predictions(args.save_preds, ids, labels)
    print(f"[INFO] save classify result: {args.save_preds}", flush=True)

    id2q = {str(q[id_field]): q for q in questions}
    for qid, lab in zip(ids, labels):
        strat = LABEL_TO_STRATEGY[lab]
        if str(qid) in id2q:
            buckets[strat].append(id2q[str(qid)])

    del clf_mod, clf_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print({k: len(v) for k, v in buckets.items()})
        
    retriever = create_retriever(args) if (len(buckets["single"]) + len(buckets["multi"]) > 0) else None
    reasoner, qa_reader, reranker = create_llm_components(args)

    if args.traces_path:
        open(args.traces_path, "w", encoding="utf-8").close()


    all_metrics = {
        "retrieval": {"em":[[],[],[]], "precision":[[],[],[]], "recall":[[],[],[]], "f1":[[],[],[]]},
        "answer": {"em":[[],[],[]], "f1":[[],[],[]], "acc":[[],[],[]]}
    }

    for strat in ["no", "single", "multi"]:
        group = buckets[strat]
        if not group:
            continue
        print(f"\n=== Running strategy={strat} on {len(group)} questions ===")
        m = run_group(strat, group, args, retriever, reranker, reasoner, qa_reader, fields)
        merge_metrics(all_metrics, m)

        print("\n[Partial metrics after this group]")
        print("Retrieval:")
        for metric in ["em","precision","recall","f1"]:
            print_metrics(all_metrics["retrieval"][metric], metric.upper())
        print("\nAnswer:")
        for metric in ["em","f1","acc"]:
            print_metrics(all_metrics["answer"][metric], metric.upper())

    print("\n===== FINAL METRICS (All questions) =====")
    print("Retrieval:")
    for metric in ["em","precision","recall","f1"]:
        print_metrics(all_metrics["retrieval"][metric], metric.upper())
    print("\nAnswer:")
    for metric in ["em","f1","acc"]:
        print_metrics(all_metrics["answer"][metric], metric.upper())

if __name__ == "__main__":
    main()
