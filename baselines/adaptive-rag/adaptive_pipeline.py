import time
import json
import random as rd
import argparse
from typing import List, Dict, Any, Union

from tqdm import tqdm
import torch
from transformers import set_seed
from vllm import LLM
import numpy as np

import os
import sys
from baselines.ircot.modules import Reasoner, QAReader
from config import DEBERTA_MAX_LENGTH, DATASET_PATHS, DATASET_FIELDS
from pipeline.utils import print_metrics, compute_retrieval_metrics, compute_answer_metrics, compute_all_answer_metrics
from pipeline.modules import AsyncOpenAIConfig
from pipeline.contriever import Retriever
from pipeline.bm25.bm25_retriever import BM25Retriever
from pipeline.verifier import Reranker


class BaseRetrieval:
    @staticmethod
    def _compute_and_log_metrics(questions, batch_history, final_traces, final_predictions, stop_logs, fields, traces_path, stop_log_path):
        em_list, precision_list, recall_list, f1_list = compute_retrieval_metrics(
            questions, batch_history, stop_logs, fields
        )
        ans_em_list, ans_f1_list, ans_acc_list = compute_answer_metrics(
            questions, final_predictions, fields
        )

        metrics = {
            "retrieval": {
                "em": em_list,
                "precision": precision_list,
                "recall": recall_list,
                "f1": f1_list,
            },
            "answer": {
                "em": ans_em_list,
                "f1": ans_f1_list,
                "acc": ans_acc_list,
            }
        }

        if stop_log_path:
            with open(stop_log_path, 'a', encoding='utf-8') as f:
                for log in stop_logs:
                    f.write(json.dumps(log, ensure_ascii=False) + '\n')

        if traces_path:
            BaseRetrieval._save_traces(
                questions, batch_history, final_traces, final_predictions, 
                traces_path, fields
            )

        return metrics

    @staticmethod
    def _save_traces(questions, batch_history, final_traces, final_predictions, traces_path, fields):
        all_ans_em_list, all_ans_f1_list, all_ans_acc_list = compute_all_answer_metrics(
            questions, final_predictions, fields
        )
        
        batch_history_indices = [[len(hist)] for hist in batch_history]
        
        with open(traces_path, 'a', encoding='utf-8') as f:
            for q, hist, hist_idx, tr, pred, em, f1, acc in zip(
                questions, batch_history, batch_history_indices,
                final_traces, final_predictions, all_ans_em_list, all_ans_f1_list, all_ans_acc_list
            ):
                info = {
                    "question_id": q[fields["id"]],
                    "question": q.get(fields["question"], ""),
                    "gold_hop": len(q.get(fields.get("supporting_facts", ""), [])) if fields.get("supporting_facts") else 0,
                    "answer": q.get(fields["answer"], ""),
                    "answer_aliases": q.get(fields.get("answer_aliases", ""), []),
                    "trace": tr,
                    "history": hist,
                    "history_indices": hist_idx,
                    "prediction": pred,
                    "em": em,
                    "f1": f1,
                    "acc": acc,
                }
                f.write(json.dumps(info, ensure_ascii=False) + "\n")

    @staticmethod
    def _create_stop_logs(questions, stop_iter, fields):
        return [{
            "question_id": q[fields["id"]],
            "gold_hop": len(q.get(fields.get("supporting_facts", ""), [])) if fields.get("supporting_facts") else 0,
            "stop_iter": stop_iter,
        } for q in questions]

    @staticmethod
    def _generate_rationales(reasoner, questions, batch_history, traces, fields):
        if reasoner is not None:
            return reasoner.batch_generate_rationales(questions, batch_history, traces, fields)
        return ["" for _ in questions]

    @staticmethod
    def _generate_answers(qa_reader, questions, batch_history, fields):
        if qa_reader is not None:
            return qa_reader.batch_generate_answers(questions, batch_history, fields)
        return [""] * len(questions)

    @staticmethod
    def _log_trace(questions, batch_history, final_traces, fields, iteration=None):
        title = f"Iteration {iteration}" if iteration else "Processing"
        print(f"\n========== {title} ==========", flush=True)
        
        for q, hist, rat in zip(questions, batch_history, final_traces):
            qid = q.get(fields["id"], "N/A")
            print(f"[TRACE] QID={qid}", flush=True)
            print(f"|  Question: {q.get('question', 'N/A')}", flush=True)
            for d in hist:
                print(f"|  Document: {d['title']}: {d['text'][:100]}...", flush=True)
            print(f"|  Generated Rationale: {rat}\n", flush=True)


class NoRetrieval(BaseRetrieval):
    @staticmethod
    def run_batch(reasoner, qa_reader, questions, max_search, search_batch_size, traces_path, log_trace, fields):
        batch_history = [[] for _ in range(len(questions))]
        empty_traces = ["" for _ in questions]
        
        final_traces = BaseRetrieval._generate_rationales(
            reasoner, questions, batch_history, empty_traces, fields
        )
        final_traces = [r.strip() for r in final_traces]

        if log_trace:
            BaseRetrieval._log_trace(questions, batch_history, final_traces, fields)

        final_predictions = BaseRetrieval._generate_answers(
            qa_reader, questions, batch_history, fields
        )

        stop_logs = BaseRetrieval._create_stop_logs(questions, 0, fields)
        
        return BaseRetrieval._compute_and_log_metrics(
            questions, batch_history, final_traces, final_predictions,
            stop_logs, fields, traces_path, None
        )


class SingleRetrieval(BaseRetrieval):
    @staticmethod
    def run_batch(retriever, reranker, reasoner, qa_reader, questions, max_search, search_batch_size, max_docs, allow_duplicate_docs, traces_path, stop_log_path, log_trace, fields):
        start_time = time.time()

        search_queries = [q[fields["question"]] for q in questions]
        batch_docs = SingleRetrieval._batch_retrieve(
            retriever, search_queries, max_search, search_batch_size
        )

        batch_scores = SingleRetrieval._batch_rerank(
            reranker, search_queries, batch_docs
        )

        batch_history = SingleRetrieval._select_documents(
            batch_docs, batch_scores, max_docs, allow_duplicate_docs
        )

        traces_in = ["" for _ in questions]
        final_traces = BaseRetrieval._generate_rationales(
            reasoner, questions, batch_history, traces_in, fields
        )
        final_traces = [r.strip() for r in final_traces]

        if log_trace:
            BaseRetrieval._log_trace(questions, batch_history, final_traces, fields)

        final_predictions = BaseRetrieval._generate_answers(
            qa_reader, questions, batch_history, fields
        )

        stop_logs = BaseRetrieval._create_stop_logs(questions, 1, fields)
        
        print(f"Single-retrieval completed in {time.time() - start_time:.2f} seconds", flush=True)
        
        return BaseRetrieval._compute_and_log_metrics(
            questions, batch_history, final_traces, final_predictions,
            stop_logs, fields, traces_path, stop_log_path
        )

    @staticmethod
    def _batch_retrieve(retriever, search_queries, max_search, search_batch_size):
        batch_docs = []
        for i in range(0, len(search_queries), search_batch_size):
            batch_queries = search_queries[i:i + search_batch_size]
            docs_list = retriever.search(batch_queries, max_search)
            batch_docs.extend(docs_list)
        return batch_docs

    @staticmethod
    def _batch_rerank(reranker, search_queries, batch_docs):
        if reranker is not None:
            return reranker.batch_rank(search_queries, batch_docs)
        return [np.array([doc['score'] for doc in docs], dtype=float) for docs in batch_docs]

    @staticmethod
    def _select_documents(batch_docs, batch_scores, max_docs, allow_duplicate_docs):
        batch_history = []
        for docs, scores in zip(batch_docs, batch_scores):
            if not allow_duplicate_docs:
                seen = set()
                for j, d in enumerate(docs):
                    if d["id"] in seen:
                        scores[j] = float("-inf")
                    else:
                        seen.add(d["id"])

            order = np.argsort(scores)[::-1]
            selected = [docs[j] for j in order[:max_docs]]
            batch_history.append(selected)
        return batch_history


class MultiRetrieval(BaseRetrieval):
    @staticmethod
    def run_batch(retriever, reranker, reasoner, qa_reader, questions, max_iterations, max_search, search_batch_size, max_docs, allow_duplicate_docs, traces_path, stop_log_path, log_trace, fields):
        final_questions = []
        final_batch_history = []
        final_traces = []
        final_predictions = []

        batch_history = [[] for _ in range(len(questions))]
        batch_history_indices = [[] for _ in range(len(questions))]
        traces = ["" for _ in questions]
        search_queries = [question[fields["question"]] for question in questions]
        
        iter_count = 0
        stop_logs = []

        while questions:
            start_time = time.time()
            iter_count += 1

            batch_docs = SingleRetrieval._batch_retrieve(
                retriever, search_queries, max_search, search_batch_size
            )
            batch_scores = SingleRetrieval._batch_rerank(
                reranker, search_queries, batch_docs
            )

            MultiRetrieval._update_histories(
                batch_history, batch_history_indices, batch_docs, batch_scores, 
                max_docs, allow_duplicate_docs
            )

            rationales = BaseRetrieval._generate_rationales(
                reasoner, questions, batch_history, traces, fields
            )

            if log_trace:
                print(f"\n========== Iteration {iter_count} ==========", flush=True)
                for question, trace, rationale in zip(questions, traces, rationales):
                    print(f"[TRACE] QID={question[fields['id']]}", flush=True)
                    print(f"|  Question: {question[fields['question']]}", flush=True)
                    print(f"|  Previous Trace: {trace}", flush=True)
                    print(f"|  Generated Rationale: {rationale}\n", flush=True)

            questions, batch_history, batch_history_indices, traces, search_queries = (
                MultiRetrieval._process_iteration_results(
                    questions, batch_history, batch_history_indices, traces, 
                    rationales, iter_count, max_iterations, stop_logs, fields,
                    final_questions, final_batch_history, final_traces
                )
            )

            print(f"Iteration {iter_count} completed in {time.time() - start_time:.2f} seconds", flush=True)
            print(f"Remaining: {len(questions)}, Completed: {len(final_questions)}\n", flush=True)

        final_predictions = BaseRetrieval._generate_answers(
            qa_reader, final_questions, final_batch_history, fields
        )

        if log_trace:
            MultiRetrieval._log_final_results(
                final_questions, final_batch_history, final_predictions, fields
            )

        return BaseRetrieval._compute_and_log_metrics(
            final_questions, final_batch_history, final_traces, final_predictions,
            stop_logs, fields, traces_path, stop_log_path
        )

    @staticmethod
    def _update_histories(batch_history, batch_history_indices, batch_docs, batch_scores, max_docs, allow_duplicate_docs):
        for history, history_indices, docs, scores in zip(
            batch_history, batch_history_indices, batch_docs, batch_scores
        ):
            if not allow_duplicate_docs:
                for i, doc in enumerate(docs):
                    if doc["id"] in {d["id"] for d in history}:
                        scores[i] = float("-inf")

            sorted_docs = [docs[i] for i in np.argsort(scores)[::-1]]
            selected_docs = sorted_docs[:max_docs]

            for doc in selected_docs:
                if doc["id"] not in {d["id"] for d in history}:
                    history.append(doc)
            history_indices.append(len(history))

    @staticmethod
    def _process_iteration_results(questions, batch_history, batch_history_indices, traces, rationales, iter_count, max_iterations, stop_logs, fields, final_questions, final_batch_history, final_traces):
        next_questions = []
        next_batch_history = []
        next_batch_history_indices = []
        next_traces = []
        search_queries = []

        for question, history, history_indices, trace, rationale in zip(
            questions, batch_history, batch_history_indices, traces, rationales
        ):
            trace += " " + rationale

            if "answer is" in rationale.lower() or iter_count >= max_iterations:
                final_questions.append(question)
                final_batch_history.append(history)
                final_traces.append(trace)

                stop_logs.append({
                    "question_id": question[fields["id"]],
                    "gold_hop": len(question.get(fields.get("supporting_facts", ""), [])) if fields.get("supporting_facts") else 0,
                    "stop_iter": iter_count
                })
            else:
                next_questions.append(question)
                next_batch_history.append(history)
                next_batch_history_indices.append(history_indices)
                next_traces.append(trace)
                search_queries.append(rationale)

        return (next_questions, next_batch_history, next_batch_history_indices, 
                next_traces, search_queries)

    @staticmethod
    def _log_final_results(final_questions, final_batch_history, final_predictions, fields):
        print("\n========== Final Results ==========", flush=True)
        for question, history, prediction in zip(final_questions, final_batch_history, final_predictions):
            print(f"Question: {question[fields['question']]}", flush=True)
            print("History:", flush=True)
            for doc in history:
                print(f"|  {doc['title']}: {doc['text']}", flush=True)
            print(f"Prediction: {prediction}", flush=True)
            print(f"Answer: {question[fields['answer']]}\n", flush=True)


def create_retriever(args):
    passages = args.passages or DATASET_PATHS[args.dataset]["passages"]

    if args.retriever_type == "contriever":
        embeddings = args.contriever_embeddings or DATASET_PATHS[args.dataset]["contriever_embeddings"]
        return Retriever(
            passages, embeddings,
            model_type="contriever",
            model_path="facebook/contriever-msmarco",
        )
    elif args.retriever_type == "bm25":
        index_path_dir = args.bm25_index_path_dir or DATASET_PATHS[args.dataset]["bm25_index"]
        return BM25Retriever(
            passages, index_path_dir=index_path_dir,
            save_or_load_index=True, k1=args.bm25_k1, b=args.bm25_b,
            method=args.bm25_method, use_mmap=args.bm25_use_mmap,
        )
    else:
        raise ValueError(f"Unknown retriever type: {args.retriever_type}")


def create_llm_components(args):
    vllm_agent = None
    if "vllm" in [args.reasoner_provider, args.qareader_provider]:
        vllm_agent = LLM(
            model=args.vllm_model_id,
            tensor_parallel_size=args.vllm_tp_size,
            quantization=args.vllm_quantization,
            dtype=torch.bfloat16,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=args.vllm_max_model_len,
        )

    openai_config = None
    if "openai" in [args.reasoner_provider, args.qareader_provider]:
        openai_config = AsyncOpenAIConfig(
            model_id=args.openai_model_id,
            max_retries=args.openai_max_retries,
            batch_timeout=args.openai_batch_timeout,
            total_timeout=args.openai_total_timeout,
            connect_timeout=args.openai_connect_timeout,
            max_keepalive_connections=args.openai_max_keepalive_connections,
            max_connections=args.openai_max_connections,
            max_concurrent=args.openai_max_concurrent,
        )

    with open(args.icl_examples_path, "r", encoding="utf-8") as f:
        icl_examples = f.read()
        icl_examples = "\n".join([line for line in icl_examples.split("\n") 
                                 if not line.startswith("# METADATA")])

    reasoner = Reasoner(
        llm=vllm_agent if args.reasoner_provider == "vllm" else openai_config,
        icl_examples=icl_examples,
        max_gen_length=args.reasoner_max_gen_length,
        temperature=args.reasoner_temperature,
        top_p=args.reasoner_top_p,
        provider=args.reasoner_provider,
    )

    qa_reader = None
    if not args.qareader_disable:
        qa_reader = QAReader(
            llm=vllm_agent if args.qareader_provider == "vllm" else openai_config,
            icl_examples=icl_examples,
            max_gen_length=args.qareader_max_gen_length,
            temperature=args.qareader_temperature,
            top_p=args.qareader_top_p,
            provider=args.qareader_provider,
        )

    reranker = None
    if not args.reranker_disable:
        reranker = Reranker(
            model_id=args.reranker_model_id,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
        )

    return reasoner, qa_reader, reranker


def load_dataset(args):
    dataset_path = args.dataset_path or DATASET_PATHS[args.dataset][args.dataset_type]
    with open(dataset_path, "r", encoding="utf-8") as f:
        if args.dataset == "musique":
            questions = [json.loads(line.strip()) for line in f]
        else:
            questions = json.load(f)
        rd.shuffle(questions)
    return questions


def run_strategy(args, strategy_class, **kwargs):
    if args.strategy == "no":
        return strategy_class.run_batch(**kwargs)
    elif args.strategy == "single":
        return strategy_class.run_batch(**kwargs)
    else:  # multi
        return strategy_class.run_batch(max_iterations=args.max_iterations, **kwargs)


def process_batches(args, questions, retriever, reasoner, qa_reader, reranker):
    strategy_map = {
        "no": NoRetrieval,
        "single": SingleRetrieval, 
        "multi": MultiRetrieval
    }
    
    all_metrics = {
        "retrieval": {"em": [[], [], []], "precision": [[], [], []], 
                     "recall": [[], [], []], "f1": [[], [], []]},
        "answer": {"em": [[], [], []], "f1": [[], [], []], "acc": [[], [], []]}
    }

    total_batches = (len(questions) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch_questions = questions[i:i + args.batch_size]
        print(f"\nProcessing batch {i // args.batch_size + 1} of {total_batches}...\n", flush=True)

        common_args = {
            "reasoner": reasoner,
            "qa_reader": qa_reader,
            "questions": batch_questions,
            "max_search": args.max_search,
            "search_batch_size": args.search_batch_size,
            "traces_path": args.traces_path,
            "log_trace": args.log_trace,
            "fields": DATASET_FIELDS[args.dataset],
        }

        if args.strategy != "no":
            common_args.update({
                "retriever": retriever,
                "reranker": reranker,
                "max_docs": args.max_docs,
                "allow_duplicate_docs": args.allow_duplicate_docs,
                "stop_log_path": args.stop_log_path,
            })

        strategy_class = strategy_map[args.strategy]
        batch_metrics = run_strategy(args, strategy_class, **common_args)

        for metric_type in ["retrieval", "answer"]:
            for metric_name in batch_metrics[metric_type]:
                for hop_idx in range(3):
                    all_metrics[metric_type][metric_name][hop_idx].extend(
                        batch_metrics[metric_type][metric_name][hop_idx]
                    )

        print("\n===== CUMULATIVE METRICS =====")
        print("Retrieval:")
        for metric in ["em", "precision", "recall", "f1"]:
            print_metrics(all_metrics["retrieval"][metric], metric.upper())
        
        print("\nAnswer:")
        for metric in ["em", "f1", "acc"]:
            print_metrics(all_metrics["answer"][metric], metric.upper())

    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser()

    retriever_group = parser.add_argument_group("Retriever")
    retriever_group.add_argument("--passages", type=str)
    retriever_group.add_argument("--retriever-type", type=str, default="contriever", 
                                choices=["contriever", "bm25"])
    retriever_group.add_argument("--max-search", type=int, default=10)
    retriever_group.add_argument("--search-batch-size", type=int, default=16)
    retriever_group.add_argument("--max-docs", type=int, default=1)
    retriever_group.add_argument("--allow-duplicate-docs", action="store_true")
    retriever_group.add_argument("--contriever-embeddings", type=str)
    retriever_group.add_argument("--bm25-index-path-dir", type=str)
    retriever_group.add_argument("--bm25-k1", type=float, default=1.5)
    retriever_group.add_argument("--bm25-b", type=float, default=0.8)
    retriever_group.add_argument("--bm25-method", type=str, default="lucene", 
                                choices=["lucene", "robertson", "atire", "bm25l", "bm25+"])
    retriever_group.add_argument("--bm25-use-mmap", action="store_true")

    vllm_group = parser.add_argument_group("vLLM")
    vllm_group.add_argument("--vllm-model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    vllm_group.add_argument("--vllm-tp-size", type=int, default=1)
    vllm_group.add_argument("--vllm-quantization", type=str)
    vllm_group.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    vllm_group.add_argument("--vllm-max-model-len", type=int)

    openai_group = parser.add_argument_group("OpenAI")
    openai_group.add_argument("--openai-model-id", type=str, default="gpt-4o-mini-2024-07-18")
    openai_group.add_argument("--openai-max-retries", type=int, default=1)
    openai_group.add_argument("--openai-batch-timeout", type=int, default=600)
    openai_group.add_argument("--openai-total-timeout", type=int, default=60)
    openai_group.add_argument("--openai-connect-timeout", type=int, default=10)
    openai_group.add_argument("--openai-max-keepalive-connections", type=int, default=10)
    openai_group.add_argument("--openai-max-connections", type=int, default=20)
    openai_group.add_argument("--openai-max-concurrent", type=int, default=3)

    reasoner_group = parser.add_argument_group("Reasoner")
    reasoner_group.add_argument("--reasoner-max-gen-length", type=int, default=400)
    reasoner_group.add_argument("--reasoner-temperature", type=float, default=0.0)
    reasoner_group.add_argument("--reasoner-top-p", type=float, default=1.0)
    reasoner_group.add_argument("--reasoner-provider", type=str, default="vllm", 
                               choices=["vllm", "openai"])

    qa_group = parser.add_argument_group("QA Reader")
    qa_group.add_argument("--qareader-max-gen-length", type=int, default=400)
    qa_group.add_argument("--qareader-temperature", type=float, default=0.0)
    qa_group.add_argument("--qareader-top-p", type=float, default=1.0)
    qa_group.add_argument("--qareader-provider", type=str, default="vllm", 
                         choices=["vllm", "openai"])
    qa_group.add_argument("--qareader-disable", action="store_true")

    reranker_group = parser.add_argument_group("Reranker")
    reranker_group.add_argument("--reranker-model-id", type=str, default="BAAI/bge-reranker-v2-m3")
    reranker_group.add_argument("--reranker-batch-size", type=int, default=8)
    reranker_group.add_argument("--reranker-max-length", type=int, default=DEBERTA_MAX_LENGTH)
    reranker_group.add_argument("--reranker-disable", action="store_true")

    main_group = parser.add_argument_group("Main")
    main_group.add_argument("--dataset", type=str, required=True, choices=DATASET_PATHS.keys())
    main_group.add_argument("--dataset-type", type=str, required=True, choices=["train", "dev", "test", "eval_subsampled", "test_subsampled", "adaptive-rag_subsampled"])
    main_group.add_argument("--dataset-path", type=str)
    main_group.add_argument("--icl-examples-path", type=str, required=True)
    main_group.add_argument("--batch-size", type=int, default=32)
    main_group.add_argument("--max-iterations", type=int, default=10)
    main_group.add_argument("--traces-path", type=str)
    main_group.add_argument("--stop-log-path", type=str)
    main_group.add_argument("--log-trace", action="store_true")
    main_group.add_argument("--seed", type=int, default=42)
    main_group.add_argument("--strategy", type=str, required=True, choices=["no", "single", "multi"])

    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    
    retriever = create_retriever(args) if args.strategy != "no" else None
    reasoner, qa_reader, reranker = create_llm_components(args)
    questions = load_dataset(args)

    if args.traces_path:
        open(args.traces_path, "w", encoding="utf-8").close()

    all_metrics = process_batches(args, questions, retriever, reasoner, qa_reader, reranker)

    print("\n===== FINAL METRICS =====")
    print("Retrieval:")
    for metric in ["em", "precision", "recall", "f1"]:
        print_metrics(all_metrics["retrieval"][metric], metric.upper())
    
    print("\nAnswer:")
    for metric in ["em", "f1", "acc"]:
        print_metrics(all_metrics["answer"][metric], metric.upper())

    print("\nCompleted successfully!")


if __name__ == "__main__":
    args = parse_args()
    main(args)