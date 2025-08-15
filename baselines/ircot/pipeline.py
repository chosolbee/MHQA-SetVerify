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

from .modules import Reasoner, QAReader
from config import DEBERTA_MAX_LENGTH, DATASET_PATHS, DATASET_FIELDS
from pipeline.utils import print_metrics, compute_retrieval_metrics, compute_answer_metrics, compute_all_answer_metrics
from pipeline.modules import AsyncOpenAIConfig
from pipeline.contriever import Retriever
from pipeline.bm25.bm25_retriever import BM25Retriever
from pipeline.verifier import Reranker


def run_batch(
    retriever: Union[Retriever, BM25Retriever],
    reranker: Reranker,
    reasoner: Reasoner,
    qa_reader: QAReader,
    questions: List[Dict[str, Any]],
    max_iterations: int,
    max_search: int,
    search_batch_size: int,
    max_docs: int,
    allow_duplicate_docs: bool,
    traces_path: str,
    stop_log_path: str,
    log_trace: bool,
    fields: Dict[str, str],
) -> Dict[str, List[List[float]]]:
    final_questions = []
    final_batch_history = []
    final_batch_history_indices = []
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

        batch_docs = []
        for i in range(0, len(search_queries), search_batch_size):
            batch_queries = search_queries[i:i + search_batch_size]
            docs = retriever.search(batch_queries, max_search)
            batch_docs.extend(docs)

        if reranker is not None:
            batch_scores = reranker.batch_rank(search_queries, batch_docs)
        else:
            batch_scores = [np.array([doc['score'] for doc in docs], dtype=float) for docs in batch_docs]

        batch_selected_docs = []
        for history, history_indices, trace, docs, scores in zip(
            batch_history, batch_history_indices, traces, batch_docs, batch_scores
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
            batch_selected_docs.append(selected_docs)

        rationales = reasoner.batch_generate_rationales(questions, batch_history, traces, fields)

        if log_trace:
            print(f"\n========== Iteration {iter_count} ==========", flush=True)
            for question, trace, docs, rationale in zip(questions, traces, batch_selected_docs, rationales):
                print(f"[TRACE] QID={question[fields['id']]}", flush=True)
                print(f"|  Question: {question[fields['question']]}", flush=True)
                print(f"|  Previous Trace: {trace}", flush=True)
                for doc in docs:
                    print(f"|  Document: {doc['title']}: {doc['text'][:100]}...", flush=True)
                print(f"|  Generated Rationale: {rationale}\n", flush=True)

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
                final_batch_history_indices.append(history_indices)
                final_traces.append(trace)

                stop_logs.append({
                    "question_id": question[fields["id"]],
                    "gold_hop": len(question.get(fields["supporting_facts"], [])),
                    "stop_iter": iter_count
                })
            else:
                next_questions.append(question)
                next_batch_history.append(history)
                next_batch_history_indices.append(history_indices)
                next_traces.append(trace)
                search_queries.append(rationale)

        questions = next_questions
        batch_history = next_batch_history
        batch_history_indices = next_batch_history_indices
        traces = next_traces

        print(f"Iteration {iter_count} completed in {time.time() - start_time:.2f} seconds", flush=True)
        print(f"Remaining questions: {len(questions)}", flush=True)
        print(f"Completed questions: {len(final_questions)}\n", flush=True)

    if qa_reader is not None:
        final_predictions = qa_reader.batch_generate_answers(final_questions, final_batch_history, fields)
    else:
        final_predictions = [""] * len(final_questions)

    if log_trace:
        print("\n========== Final Questions and History ==========", flush=True)
        for question, history, prediction in zip(final_questions, final_batch_history, final_predictions):
            print(f"1. Question: {question[fields['question']]}", flush=True)
            print("2. History:", flush=True)
            for doc in history:
                print(f"|  {doc['title']}: {doc['text']}", flush=True)
            print(f"3. Prediction: {prediction}\n", flush=True)
            print(f"4. Answer: {question[fields['answer']]}", flush=True)

    em_list, precision_list, recall_list, f1_list = compute_retrieval_metrics(final_questions, final_batch_history, stop_logs, fields)
    ans_em_list, ans_f1_list, ans_acc_list = compute_answer_metrics(final_questions, final_predictions, fields)

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
        all_ans_em_list, all_ans_f1_list, all_ans_acc_list = compute_all_answer_metrics(final_questions, final_predictions, fields)
        with open(traces_path, 'a', encoding='utf-8') as f:
            for question, history, history_indices, trace, prediction, em, f1, acc in zip(
                final_questions, final_batch_history, final_batch_history_indices, final_traces, final_predictions, all_ans_em_list, all_ans_f1_list, all_ans_acc_list
            ):
                info = {
                    "question_id": question[fields["id"]],
                    "question": question.get(fields["question"], ""),
                    "gold_hop": len(question.get(fields["supporting_facts"], [])),
                    "answer": question.get(fields["answer"], ""),
                    "answer_aliases": question.get(fields["answer_aliases"], []),
                    "trace": trace.strip(),
                    "history": history,
                    "history_indices": history_indices,
                    "prediction": prediction,
                    "em": em,
                    "f1": f1,
                    "acc": acc,
                }
                f.write(json.dumps(info, ensure_ascii=False) + '\n')

    return metrics


def parse_args():
    parser = argparse.ArgumentParser()

    retriever_group = parser.add_argument_group("Retriever Options")
    retriever_group.add_argument("--passages", type=str, help="document file path")
    retriever_group.add_argument("--retriever-type", type=str, default="contriever", choices=["contriever", "bm25"], help="Type of retriever to use")
    retriever_group.add_argument("--max-search", type=int, default=10, help="Maximum number of passages to retrieve")
    retriever_group.add_argument("--search-batch-size", type=int, default=16, help="Batch size for search queries")
    retriever_group.add_argument("--max-docs", type=int, default=1, help="Maximum number of documents to select from retrieved passages")
    retriever_group.add_argument("--allow-duplicate-docs", action="store_true", help="Allow duplicate documents in the retrieved passages")
    # contriever
    retriever_group.add_argument("--contriever-embeddings", type=str, help="Document embedding path")
    # bm25
    retriever_group.add_argument("--bm25-index-path-dir", type=str, help="BM25 index path directory")
    retriever_group.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1 parameter")
    retriever_group.add_argument("--bm25-b", type=float, default=0.8, help="BM25 b parameter")
    retriever_group.add_argument("--bm25-method", type=str, default="lucene", choices=["lucene", "robertson", "atire", "bm25l", "bm25+"], help="BM25 variant method")
    retriever_group.add_argument("--bm25-use-mmap", action="store_true", help="Use memory mapping for large BM25 indices")

    vllm_group = parser.add_argument_group("vLLM Options")
    vllm_group.add_argument("--vllm-model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID for vLLM")
    vllm_group.add_argument("--vllm-tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    vllm_group.add_argument("--vllm-quantization", type=str, help="Quantization method for vLLM")
    vllm_group.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    vllm_group.add_argument("--vllm-max-model-len", type=int, help="Maximum model length for vLLM")

    openai_group = parser.add_argument_group("OpenAI Options")
    openai_group.add_argument("--openai-model-id", type=str, default="gpt-4o-mini-2024-07-18", help="Model ID for OpenAI")
    openai_group.add_argument("--openai-max-retries", type=int, default=1, help="Maximum retries for OpenAI requests")
    openai_group.add_argument("--openai-batch-timeout", type=int, default=600, help="Batch timeout for OpenAI requests")
    openai_group.add_argument("--openai-total-timeout", type=int, default=60, help="Total timeout for OpenAI requests")
    openai_group.add_argument("--openai-connect-timeout", type=int, default=10, help="Connection timeout for OpenAI requests")
    openai_group.add_argument("--openai-max-keepalive-connections", type=int, default=10, help="Maximum keepalive connections for OpenAI")
    openai_group.add_argument("--openai-max-connections", type=int, default=20, help="Maximum connections for OpenAI")
    openai_group.add_argument("--openai-max-concurrent", type=int, default=3, help="Maximum concurrent requests for OpenAI")

    reasoner_group = parser.add_argument_group("Reasoner Options")
    reasoner_group.add_argument("--reasoner-max-gen-length", type=int, default=400, help="Maximum generation length for reasoner")
    reasoner_group.add_argument("--reasoner-temperature", type=float, default=0.0, help="Temperature for reasoner")
    reasoner_group.add_argument("--reasoner-top-p", type=float, default=1.0, help="Top-p sampling for reasoner")
    reasoner_group.add_argument("--reasoner-provider", type=str, default="vllm", choices=["vllm", "openai"], help="Provider for reasoner")

    qa_reader_group = parser.add_argument_group("QA Reader Options")
    qa_reader_group.add_argument("--qareader-max-gen-length", type=int, default=400, help="Maximum generation length for QA reader")
    qa_reader_group.add_argument("--qareader-temperature", type=float, default=0.0, help="Temperature for QA reader")
    qa_reader_group.add_argument("--qareader-top-p", type=float, default=1.0, help="Top-p sampling for QA reader")
    qa_reader_group.add_argument("--qareader-provider", type=str, default="vllm", choices=["vllm", "openai"], help="Provider for QA reader")
    qa_reader_group.add_argument("--qareader-disable", action="store_true", help="Disable QA reader")

    reranker_group = parser.add_argument_group("Reranker Options")
    reranker_group.add_argument("--reranker-model-id", type=str, default="BAAI/bge-reranker-v2-m3", help="Model ID for reranker")
    reranker_group.add_argument("--reranker-batch-size", type=int, default=8, help="Batch size for reranker")
    reranker_group.add_argument("--reranker-max-length", type=int, default=DEBERTA_MAX_LENGTH, help="Maximum length for reranker input")
    reranker_group.add_argument("--reranker-disable", action="store_true", help="Disable reranker and use retriever scores")

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--dataset", type=str, required=True, choices=DATASET_PATHS.keys(), help="Dataset name")
    main_group.add_argument("--dataset-type", type=str, required=True, choices=["train", "dev", "test", "eval_subsampled", "test_subsampled"], help="Dataset type")
    main_group.add_argument("--dataset-path", type=str, help="Dataset file path")
    main_group.add_argument("--icl-examples-path", type=str, required=True, help="Path to ICL examples")
    main_group.add_argument("--batch-size", type=int, default=32, help="Batch size for processing questions")
    main_group.add_argument("--max-iterations", type=int, default=10, help="Maximum number of iterations")
    main_group.add_argument("--traces-path", type=str, help="Path to save traces")
    main_group.add_argument("--stop-log-path", type=str, default=None, help="Optional JSONL path; Path to the JSONL file where stopping logs are written")
    main_group.add_argument("--log-trace", action="store_true", help="Enable detailed trace logging")
    main_group.add_argument("--seed", type=int, default=42, help="Random Seed")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    set_seed(args.seed)

    passages = args.passages or DATASET_PATHS[args.dataset]["passages"]

    if args.retriever_type == "contriever":
        embeddings = args.contriever_embeddings or DATASET_PATHS[args.dataset]["contriever_embeddings"]
        retriever = Retriever(
            passages,
            embeddings,
            model_type="contriever",
            model_path="facebook/contriever-msmarco",
        )
    elif args.retriever_type == "bm25":
        index_path_dir = args.bm25_index_path_dir or DATASET_PATHS[args.dataset]["bm25_index"]
        retriever = BM25Retriever(
            passages,
            index_path_dir=index_path_dir,
            save_or_load_index=True,
            k1=args.bm25_k1,
            b=args.bm25_b,
            method=args.bm25_method,
            use_mmap=args.bm25_use_mmap,
        )
    else:
        raise ValueError(f"Unknown retriever type: {args.retriever_type}")

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
        icl_examples = "\n".join([line for line in icl_examples.split("\n") if not line.startswith("# METADATA")])

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

    if args.traces_path:
        open(args.traces_path, "w", encoding="utf-8").close()

    dataset_path = args.dataset_path or DATASET_PATHS[args.dataset][args.dataset_type]
    with open(dataset_path, "r", encoding="utf-8") as f:
        if args.dataset == "musique":
            questions = [json.loads(line.strip()) for line in f]
        else:
            questions = json.load(f)
        rd.shuffle(questions)

    all_metrics = {
        "retrieval": {
            "em": [[], [], []],
            "precision": [[], [], []],
            "recall": [[], [], []],
            "f1": [[], [], []],
        },
        "answer": {
            "em": [[], [], []],
            "f1": [[], [], []],
            "acc": [[], [], []],
        }
    }

    total_batches = (len(questions) + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch_questions = questions[i:i + args.batch_size]
        print(f"\nProcessing batch {i // args.batch_size + 1} of {total_batches}...\n", flush=True)

        batch_metrics = run_batch(
            retriever=retriever,
            reranker=reranker,
            reasoner=reasoner,
            qa_reader=qa_reader,
            questions=batch_questions,
            max_iterations=args.max_iterations,
            max_search=args.max_search,
            search_batch_size=args.search_batch_size,
            max_docs=args.max_docs,
            allow_duplicate_docs=args.allow_duplicate_docs,
            traces_path=args.traces_path,
            stop_log_path=args.stop_log_path,
            log_trace=args.log_trace,
            fields=DATASET_FIELDS[args.dataset],
        )

        for metric_type in ["retrieval", "answer"]:
            for metric_name in batch_metrics[metric_type]:
                for hop_idx in range(3):
                    all_metrics[metric_type][metric_name][hop_idx].extend(
                        batch_metrics[metric_type][metric_name][hop_idx]
                    )

        print("\n===== CUMULATIVE RETRIEVAL METRICS =====")
        print_metrics(all_metrics["retrieval"]["em"], "EM")
        print_metrics(all_metrics["retrieval"]["precision"], "Precision")
        print_metrics(all_metrics["retrieval"]["recall"], "Recall")
        print_metrics(all_metrics["retrieval"]["f1"], "F1")

        print("\n===== CUMULATIVE ANSWER METRICS =====")
        print_metrics(all_metrics["answer"]["em"], "EM")
        print_metrics(all_metrics["answer"]["f1"], "F1")
        print_metrics(all_metrics["answer"]["acc"], "Acc")

    print("\n===== FINAL RETRIEVAL METRICS =====")
    print_metrics(all_metrics["retrieval"]["em"], "EM")
    print_metrics(all_metrics["retrieval"]["precision"], "Precision")
    print_metrics(all_metrics["retrieval"]["recall"], "Recall")
    print_metrics(all_metrics["retrieval"]["f1"], "F1")

    print("\n===== FINAL ANSWER METRICS =====")
    print_metrics(all_metrics["answer"]["em"], "EM")
    print_metrics(all_metrics["answer"]["f1"], "F1")
    print_metrics(all_metrics["answer"]["acc"], "Acc")

    print("\nAll done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
