import os
import time
import json
import random as rd
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from vllm import LLM
import wandb

from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH
from .utils import print_metrics, compute_retrieval_metrics, compute_answer_metrics
from .contriever import Retriever
from .query_generator.query_generator_self_ask import QueryGenerator
from .verifier import Reranker, BoNVerifier
from .answer_generator import AnswerGenerator


def run_batch(retriever: Retriever,
              query_generator: QueryGenerator,
              reranker: Reranker,
              bon_verifier: BoNVerifier,
              questions: List[Dict[str, Any]],
              num_generations: int = 4,
              answer_generator: AnswerGenerator = None,
              max_iterations: int = 5,
              max_search: int = 10,
              generate_answers: bool = False,
              stop_log_path: str = None) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], Dict[str, List[List[float]]], List[str]]:
    final_questions = []
    final_batch_history = []
    final_traces = []

    batch_history = [[] for _ in range(len(questions))]
    traces = ["Question: " + question["question"] + "\n" for question in questions]
    iter_count = 0

    stop_logs = []

    while questions:
        start_time = time.time()

        traces, responses, is_query_list = query_generator.batch_generate(traces, is_first=iter_count == 0)

        search_questions = []
        search_batch_history = []
        search_traces = []
        search_queries = []

        for question, history, trace, response, is_query in zip(questions, batch_history, traces, responses, is_query_list):
            if is_query:
                search_questions.append(question)
                search_batch_history.append(history)
                search_traces.append(trace)
                search_queries.append(response)
            else:
                final_questions.append(question)
                final_batch_history.append(history)
                final_traces.append(trace)

                stop_logs.append({
                    "question_id": question["id"],
                    "gold_hop": len(question.get("question_decomposition", [])),
                    "stop_iter": iter_count + 1
                })

        search_batch_size = 16
        batch_docs = []
        for i in range(0, len(search_queries), search_batch_size):
            batch_queries = search_queries[i:i + search_batch_size]
            batch_docs.extend(retriever.search(batch_queries, max_search))
        batch_scores = reranker.batch_rank(search_queries, batch_docs)

        next_questions = []
        next_batch_history = []
        next_traces = []

        for question, history, trace, docs, scores in zip(
            search_questions, search_batch_history, search_traces, batch_docs, batch_scores
        ):
            for i, doc in enumerate(docs):
                if doc["id"] in {d["id"] for d in history}:
                    scores[i] = float("-inf")
            selected_doc = docs[scores.argmax()]
            history.append(selected_doc)

            next_questions.append(question)
            next_batch_history.append(history)
            next_traces.append(trace + f"Context: {selected_doc['text']}\n")

        questions = next_questions
        batch_history = next_batch_history
        traces = next_traces

        print(f"Iteration {iter_count+1} completed in {time.time() - start_time:.2f} seconds")
        print(f"Remaining questions: {len(questions)}\n")

        iter_count += 1
        if iter_count >= max_iterations:
            final_questions.extend(questions)
            final_batch_history.extend(batch_history)
            final_traces.extend(traces)

            for question in questions:
                stop_logs.append({
                    "question_id": question["id"],
                    "gold_hop": len(question.get("question_decomposition", [])),
                    "stop_iter": iter_count
                })
            break

    em_list, precision_list, recall_list, f1_list = compute_retrieval_metrics(final_questions, final_batch_history, stop_logs)

    if stop_log_path:
        with open(stop_log_path,'a',encoding='utf-8') as f:
            for log in stop_logs:
                f.write(json.dumps(log,ensure_ascii=False)+'\n')

    if generate_answers and answer_generator:
        indices = bon_verifier.batch_verify(final_traces, num_generations)
        final_questions = [final_questions[i] for i in indices]
        final_batch_history = [final_batch_history[i] for i in indices]
        final_predictions = answer_generator.batch_answer(final_questions, final_batch_history)
        ans_em_list, ans_f1_list = compute_answer_metrics(final_questions, final_predictions)
    else:
        final_predictions = []
        ans_em_list = [[], [], []]
        ans_f1_list = [[], [], []]

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

    return final_questions, final_batch_history, final_predictions, metrics


def parse_args():
    parser = argparse.ArgumentParser()

    retriever_group = parser.add_argument_group("Retriever Options")
    retriever_group.add_argument("--passages", type=str, required=True, help="document file path")
    retriever_group.add_argument("--embeddings", type=str, required=True, help="Document embedding path")

    query_generator_group = parser.add_argument_group("Query Generator Options")
    query_generator_group.add_argument("--qg-model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID for query generator")
    query_generator_group.add_argument("--qg-tp-size", type=int, default=1, help="Tensor parallel size for query generator")
    query_generator_group.add_argument("--qg-quantization", type=str, help="Quantization method for query generator")
    query_generator_group.add_argument("--qg-max-gen-length", type=int, default=512, help="Maximum generation length for query generator")
    query_generator_group.add_argument("--qg-temperature", type=float, default=0.7, help="Temperature for query generator")
    query_generator_group.add_argument("--qg-top-p", type=float, default=0.9, help="Top-p sampling for query generator")

    reranker_group = parser.add_argument_group("Reranker Options")
    reranker_group.add_argument("--reranker-model-id", type=str, default="BAAI/bge-reranker-v2-m3", help="Model ID for reranker")
    reranker_group.add_argument("--reranker-batch-size", type=int, default=8, help="Batch size for reranker")
    reranker_group.add_argument("--reranker-max-length", type=int, default=DEBERTA_MAX_LENGTH, help="Maximum length for reranker input")

    verifier_group = parser.add_argument_group("BoNVerifier Options")
    verifier_group.add_argument("--bon-verifier-model-id", type=str, default="microsoft/DeBERTa-v3-large", help="Model ID for verifier")
    verifier_group.add_argument("--bon-verifier-checkpoint-path", type=str, required=True, help="Checkpoint path for trained model")
    reranker_group.add_argument("--bon-verifier-batch-size", type=int, default=8, help="Batch size for verifier")
    verifier_group.add_argument("--bon-verifier-max-length", type=int, default=1024, help="Maximum length for verifier input")

    answer_generator_group = parser.add_argument_group("Answer Generator Options")
    answer_generator_group.add_argument("--generate-answers", action="store_true", help="Enable answer generation")
    answer_generator_group.add_argument("--ag-max-gen-length", type=int, default=1024, help="Maximum generation length for answer generator")
    answer_generator_group.add_argument("--ag-temperature", type=float, default=0.7, help="Temperature for answer generator")
    answer_generator_group.add_argument("--ag-top-p", type=float, default=0.9, help="Top-p sampling for answer generator")

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--questions", type=str, required=True, help="Questions file path")
    main_group.add_argument("--batch-size", type=int, default=32, help="Batch size for processing questions")
    main_group.add_argument("--num-generations", type=int, default=4, help="Number of generations per question")
    main_group.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")
    main_group.add_argument("--max-search", type=int, default=10, help="Maximum number of passages to retrieve")
    main_group.add_argument("--output-path", type=str, help="Path to save predictions and metrics")
    main_group.add_argument("--stop-log-path", type=str, default=None, help="Optional JSONL path; Path to the JSONL file where stopping logs are written")

    args = parser.parse_args()

    assert args.batch_size % args.num_generations == 0, "Number of generations must divide batch size evenly."

    return args


def main(args: argparse.Namespace):
    rd.seed(42)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in [-1, 0]:
        wandb.init(project="MultiHopQA-test", entity=WANDB_ENTITY)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    retriever = Retriever(
        args.passages,
        args.embeddings,
        model_type="contriever",
        model_path="facebook/contriever-msmarco",
    )

    shared_llm = LLM(
        model=args.qg_model_id,
        tensor_parallel_size=args.qg_tp_size,
        quantization=args.qg_quantization,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    query_generator = QueryGenerator(
        llm=shared_llm,
        max_gen_length=args.qg_max_gen_length,
        temperature=args.qg_temperature,
        top_p=args.qg_top_p,
    )

    reranker = Reranker(
        model_id=args.reranker_model_id,
        batch_size=args.batch_size,
        max_length=args.reranker_max_length,
    )

    bon_verifier = BoNVerifier(
        model_id=args.bon_verifier_model_id,
        checkpoint_path=args.bon_verifier_checkpoint_path,
        batch_size=args.bon_verifier_batch_size,
        max_length= args.bon_verifier_max_length,
    )

    if args.generate_answers:
        answer_generator = AnswerGenerator(
            llm=shared_llm,
            max_gen_length=args.ag_max_gen_length,
            temperature=args.ag_temperature,
            top_p=args.ag_top_p,
        )
    else:
        answer_generator = None

    if args.stop_log_path:
        open(args.stop_log_path, "w", encoding="utf-8").close()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(q) for q in f]
        questions = list(np.repeat(questions, args.num_generations, axis=0))

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
    for i in range(0, len(questions), args.batch_size):
        batch_questions = questions[i:i + args.batch_size]
        print(f"\nProcessing batch {i // args.batch_size + 1} of {total_batches}...\n")

        final_questions, final_batch_history, final_predictions, batch_metrics = run_batch(
            retriever=retriever,
            query_generator=query_generator,
            reranker=reranker,
            bon_verifier=bon_verifier,
            answer_generator=answer_generator,
            questions=batch_questions,
            num_generations=args.num_generations,
            max_iterations=args.max_iterations,
            max_search=args.max_search,
            generate_answers=args.generate_answers,
            stop_log_path=args.stop_log_path,
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

        print("\n===== BATCH RETRIEVAL METRICS =====")
        print_metrics(batch_metrics["retrieval"]["em"], "EM")
        print_metrics(batch_metrics["retrieval"]["precision"], "Precision")
        print_metrics(batch_metrics["retrieval"]["recall"], "Recall")
        print_metrics(batch_metrics["retrieval"]["f1"], "F1")

        if args.generate_answers:
            print("\n===== BATCH ANSWER METRICS =====")
            print_metrics(batch_metrics["answer"]["em"], "EM")
            print_metrics(batch_metrics["answer"]["f1"], "F1")

    print("\n===== FINAL RETRIEVAL METRICS =====")
    print_metrics(all_metrics["retrieval"]["em"], "EM")
    print_metrics(all_metrics["retrieval"]["precision"], "Precision")
    print_metrics(all_metrics["retrieval"]["recall"], "Recall")
    print_metrics(all_metrics["retrieval"]["f1"], "F1")

    if args.generate_answers:
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
