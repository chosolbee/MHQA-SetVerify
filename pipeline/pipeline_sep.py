import os
import time
import json
import random as rd
import argparse
from typing import List, Dict, Any, Tuple, Union

from tqdm import tqdm
import torch
from transformers import set_seed
from vllm import LLM
import wandb
import numpy as np

from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH, DATASET_PATHS, DATASET_FIELDS
from .utils import print_metrics, compute_retrieval_metrics, compute_answer_metrics, compute_all_answer_metrics
from .modules import AsyncOpenAIConfig
from .contriever import Retriever
from .bm25.bm25_retriever import BM25Retriever
from .query_generator.query_generator_sep import QueryGenerator
from .verifier import Reranker
from .answer_generator.answer_generator_sep import AnswerGenerator
from .stop_decider import StopDecider


def run_batch(
    retriever: Union[Retriever, BM25Retriever],
    query_generator: QueryGenerator,
    reranker: Reranker,
    intermediate_answer_generator: AnswerGenerator,
    final_answer_generator: AnswerGenerator,
    stop_decider: StopDecider,
    questions: List[Dict[str, Any]],
    max_iterations: int = 5,
    max_search: int = 10,
    search_batch_size: int = 16,
    max_docs: int = 1,
    allow_duplicate_docs: bool = False,
    traces_path: str = None,
    stop_log_path: str = None,
    log_trace: bool = False,
    fields: Dict[str, str] = None,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], List[str], Dict[str, List[List[float]]]]:
    final_questions = []
    final_batch_history = []
    final_batch_history_indices = []
    final_traces = []

    batch_history = [[] for _ in range(len(questions))]
    batch_history_indices = [[] for _ in range(len(questions))]
    traces = ["" for _ in questions]
    iter_count = 0

    stop_logs = []

    while questions:
        start_time = time.time()

        new_traces, search_queries = query_generator.batch_generate(questions, traces, fields)
        traces = new_traces

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
        for question, history, history_indices, trace, docs, scores in zip(
            questions, batch_history, batch_history_indices, traces, batch_docs, batch_scores
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

        inter_answers = intermediate_answer_generator.batch_generate_intermediate_answers(search_queries, batch_selected_docs)

        traces = [
            f"{trace}\n" + "\n".join(f"Document: {doc['title']}: {doc['text']}" for doc in docs) + f"\nIntermediate answer: {answer}"
            for trace, docs, answer in zip(traces, batch_selected_docs, inter_answers)
        ]
        stop_decisions = stop_decider.batch_decide(questions, traces, fields, log_trace)

        if log_trace:
            print(f"\n========== Iteration {iter_count + 1} ==========")
            for question, query, docs, answer, decision in zip(
                questions, search_queries, batch_selected_docs, inter_answers, stop_decisions
            ):
                print(f"[TRACE] QID={question[fields['id']]}")
                print(f"|  Question: {question[fields['question']]}")
                print(f"|  Query: {query}")
                for doc in docs:
                    print(f"|  Document: {doc['title']}: {doc['text'][:100]}...")
                print(f"|  Generated Intermediate Answer: {answer}\n")
                print(f"|  Stop Decision: {decision}\n")

        next_questions = []
        next_batch_history = []
        next_batch_history_indices = []
        next_traces = []

        for question, history, history_indices, trace, decision in zip(
            questions, batch_history, batch_history_indices, traces, stop_decisions
        ):
            if decision == "STOP":
                final_questions.append(question)
                final_batch_history.append(history)
                final_batch_history_indices.append(history_indices)
                final_traces.append(trace)

                stop_logs.append({
                    "question_id": question[fields["id"]],
                    "gold_hop": len(question.get(fields["supporting_facts"], [])),
                    "stop_iter": iter_count + 1
                })
            else:
                next_questions.append(question)
                next_batch_history.append(history)
                next_batch_history_indices.append(history_indices)
                next_traces.append(trace)

        questions = next_questions
        batch_history = next_batch_history
        batch_history_indices = next_batch_history_indices
        traces = next_traces

        print(f"Iteration {iter_count+1} completed in {time.time() - start_time:.2f} seconds")
        print(f"Remaining questions: {len(questions)}")
        print(f"Completed questions: {len(final_questions)}\n")

        iter_count += 1
        if iter_count >= max_iterations:
            if questions:
                print(f"Max iterations reached. {len(questions)} questions left.")

                final_questions.extend(questions)
                final_batch_history.extend(batch_history)
                final_batch_history_indices.extend(batch_history_indices)
                final_traces.extend(traces)

                for question in questions:
                    stop_logs.append({
                        "question_id": question[fields["id"]],
                        "gold_hop": len(question.get(fields["supporting_facts"], [])),
                        "stop_iter": iter_count
                    })
            break

    if final_answer_generator is not None:
        final_traces, final_predictions = final_answer_generator.batch_generate_final_answers(final_questions, final_traces, fields)
    else:
        final_predictions = [""] * len(final_questions)

    if log_trace:
        print("\n========== Final Questions and History ==========")
        for question, history, prediction in zip(final_questions, final_batch_history, final_predictions):
            print(f"1. Question: {question[fields['question']]}")
            print("2. History:")
            for doc in history:
                print(f"|  {doc['title']}: {doc['text']}")
            print(f"3. Prediction: {prediction}")
            print()

    em_list, precision_list, recall_list, f1_list = compute_retrieval_metrics(final_questions, final_batch_history, stop_logs, fields)

    if stop_log_path:
        with open(stop_log_path, 'a', encoding='utf-8') as f:
            for log in stop_logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')

    ans_em_list, ans_f1_list = compute_answer_metrics(final_questions, final_predictions, fields)

    if traces_path:
        all_ans_em_list, all_ans_f1_list = compute_all_answer_metrics(final_questions, final_predictions, fields)
        with open(traces_path, 'a', encoding='utf-8') as f:
            for question, history, history_indices, trace, prediction, em, f1 in zip(
                final_questions, final_batch_history, final_batch_history_indices, final_traces, final_predictions, all_ans_em_list, all_ans_f1_list
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
                }
                f.write(json.dumps(info, ensure_ascii=False) + '\n')

    metrics = {
        "retrieval": {
            "em": em_list,
            "precision": precision_list,
            "recall": recall_list,
            "f1": f1_list
        },
        "answer": {
            "em": ans_em_list,
            "f1": ans_f1_list
        }
    }

    # Final trace logging
    if log_trace and final_questions:
        print("\n========== Final Results ==========")
        for q, hist, trace, pred in zip(final_questions, final_batch_history, final_traces, final_predictions):
            print(f"\nQID: {q[fields['id']]}")
            print(f"Question: {q[fields['question']]}")
            print(f"Gold Answer: {q.get(fields['answer'], 'N/A')}")
            print(f"Prediction: {pred}")
            print(f"Number of Documents Retrieved: {len(hist)}")
            print("-" * 80)

    return final_questions, final_batch_history, final_predictions, metrics


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

    query_generator_group = parser.add_argument_group("Query Generator Options")
    query_generator_group.add_argument("--qg-max-gen-length", type=int, default=200, help="Maximum generation length for query generator")
    query_generator_group.add_argument("--qg-temperature", type=float, default=0.0, help="Temperature for query generator")
    query_generator_group.add_argument("--qg-top-p", type=float, default=1.0, help="Top-p sampling for query generator")
    query_generator_group.add_argument("--qg-provider", type=str, default="vllm", choices=["vllm", "openai"], help="Provider for query generator")

    reranker_group = parser.add_argument_group("Reranker Options")
    reranker_group.add_argument("--reranker-model-id", type=str, default="BAAI/bge-reranker-v2-m3", help="Model ID for reranker")
    reranker_group.add_argument("--reranker-batch-size", type=int, default=8, help="Batch size for reranker")
    reranker_group.add_argument("--reranker-max-length", type=int, default=DEBERTA_MAX_LENGTH, help="Maximum length for reranker input")
    reranker_group.add_argument("--reranker-disable", action="store_true", help="Disable reranker and use retriever scores")

    intermediate_answer_generator_group = parser.add_argument_group("Intermediate Answer Generator Options")
    intermediate_answer_generator_group.add_argument("--iag-max-gen-length", type=int, default=400, help="Maximum generation length for answer generator")
    intermediate_answer_generator_group.add_argument("--iag-temperature", type=float, default=0.0, help="Temperature for answer generator")
    intermediate_answer_generator_group.add_argument("--iag-top-p", type=float, default=1.0, help="Top-p sampling for answer generator")
    intermediate_answer_generator_group.add_argument("--iag-provider", type=str, default="vllm", choices=["vllm", "openai"], help="Provider for answer generator")

    final_answer_generator_group = parser.add_argument_group("Final Answer Generator Options")
    final_answer_generator_group.add_argument("--fag-max-gen-length", type=int, default=400, help="Maximum generation length for answer generator")
    final_answer_generator_group.add_argument("--fag-temperature", type=float, default=0.0, help="Temperature for answer generator")
    final_answer_generator_group.add_argument("--fag-top-p", type=float, default=1.0, help="Top-p sampling for answer generator")
    final_answer_generator_group.add_argument("--fag-provider", type=str, default="vllm", choices=["vllm", "openai"], help="Provider for answer generator")
    final_answer_generator_group.add_argument("--fag-disable", action="store_true", help="Disable final answer generation")

    stop_decider_group = parser.add_argument_group("Stop Decider Options")
    stop_decider_group.add_argument("--sd-max-gen-length", type=int, default=200, help="Maximum generation length for stop decider")
    stop_decider_group.add_argument("--sd-temperature", type=float, default=0.0, help="Temperature for stop decider")
    stop_decider_group.add_argument("--sd-top-p", type=float, default=1.0, help="Top-p sampling for stop decider")
    stop_decider_group.add_argument("--sd-provider", type=str, default="vllm", choices=["vllm", "openai", "nostop"], help="Provider for stop decider")

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--dataset", type=str, required=True, choices=DATASET_PATHS.keys(), help="Dataset name")
    main_group.add_argument("--dataset-type", type=str, required=True, choices=["train", "dev", "test"], help="Dataset type")
    main_group.add_argument("--dataset-path", type=str, help="Dataset file path")
    main_group.add_argument("--batch-size", type=int, default=32, help="Batch size for processing questions")
    main_group.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")
    main_group.add_argument("--traces-path", type=str, help="Path to save traces")
    main_group.add_argument("--output-path", type=str, help="Path to save predictions and metrics")
    main_group.add_argument("--stop-log-path", type=str, default=None, help="Optional JSONL path; Path to the JSONL file where stopping logs are written")
    main_group.add_argument("--log-trace", action="store_true", help="Enable detailed trace logging")
    main_group.add_argument("--seed", type=int, default=42, help="Random Seed")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in [-1, 0]:
        wandb.init(project="MultiHopQA-test", entity=WANDB_ENTITY, config=args)
    else:
        os.environ["WANDB_MODE"] = "disabled"

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
    if "vllm" in [args.qg_provider, args.iag_provider, args.fag_provider, args.sd_provider]:
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
    if "openai" in [args.qg_provider, args.iag_provider, args.fag_provider, args.sd_provider]:
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

    query_generator = QueryGenerator(
        llm=vllm_agent if args.qg_provider == "vllm" else openai_config,
        max_gen_length=args.qg_max_gen_length,
        temperature=args.qg_temperature,
        top_p=args.qg_top_p,
        provider=args.qg_provider,
        retriever_type=args.retriever_type,
    )

    reranker = None
    if not args.reranker_disable:
        reranker = Reranker(
            model_id=args.reranker_model_id,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
        )

    intermediate_answer_generator = AnswerGenerator(
        llm=vllm_agent if args.iag_provider == "vllm" else openai_config,
        max_gen_length=args.iag_max_gen_length,
        temperature=args.iag_temperature,
        top_p=args.iag_top_p,
        provider=args.iag_provider,
    )

    final_answer_generator = None
    if not args.fag_disable:
        final_answer_generator = AnswerGenerator(
            llm=vllm_agent if args.fag_provider == "vllm" else openai_config,
            max_gen_length=args.fag_max_gen_length,
            temperature=args.fag_temperature,
            top_p=args.fag_top_p,
            provider=args.fag_provider,
        )

    stop_decider = StopDecider(
        llm=vllm_agent if args.sd_provider == "vllm" else (openai_config if args.sd_provider == "openai" else None),
        max_gen_length=args.sd_max_gen_length,
        temperature=args.sd_temperature,
        top_p=args.sd_top_p,
        provider=args.sd_provider,
    )

    if args.traces_path:
        open(args.traces_path, "w", encoding="utf-8").close()

    if args.stop_log_path:
        open(args.stop_log_path, "w", encoding="utf-8").close()

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
            "f1": [[], [], []]
        },
        "answer": {
            "em": [[], [], []],
            "f1": [[], [], []]
        }
    }

    all_final_predictions = []
    all_final_questions = []
    all_final_batch_history = []

    total_batches = (len(questions) + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch_questions = questions[i:i + args.batch_size]
        print(f"\nProcessing batch {i // args.batch_size + 1} of {total_batches}...\n")

        final_questions, final_batch_history, final_predictions, batch_metrics = run_batch(
            retriever=retriever,
            query_generator=query_generator,
            reranker=reranker,
            intermediate_answer_generator=intermediate_answer_generator,
            final_answer_generator=final_answer_generator,
            stop_decider=stop_decider,
            questions=batch_questions,
            max_iterations=args.max_iterations,
            max_search=args.max_search,
            search_batch_size=args.search_batch_size,
            max_docs=args.max_docs,
            traces_path=args.traces_path,
            stop_log_path=args.stop_log_path,
            log_trace=args.log_trace,
            fields=DATASET_FIELDS[args.dataset],
        )

        all_final_questions.extend(final_questions)
        all_final_batch_history.extend(final_batch_history)
        all_final_predictions.extend(final_predictions)

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

    print("\n===== FINAL RETRIEVAL METRICS =====")
    print_metrics(all_metrics["retrieval"]["em"], "EM")
    print_metrics(all_metrics["retrieval"]["precision"], "Precision")
    print_metrics(all_metrics["retrieval"]["recall"], "Recall")
    print_metrics(all_metrics["retrieval"]["f1"], "F1")

    print("\n===== FINAL ANSWER METRICS =====")
    print_metrics(all_metrics["answer"]["em"], "EM")
    print_metrics(all_metrics["answer"]["f1"], "F1")

    if args.output_path:
        output_data = {
            "metrics": all_metrics,
            "predictions": []
        }

        for q, h, p in zip(all_final_questions, all_final_batch_history, all_final_predictions):
            output_data["predictions"].append({
                "id": q["id"],
                "question": q["question"],
                "gold_answer": q.get("answer", ""),
                "prediction": p if p else "",
                "passages": [doc["text"] for doc in h]
            })

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\nAll done!")
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
