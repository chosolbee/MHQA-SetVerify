import os
import time
import json
import random as rd
import argparse
from typing import List, Dict, Any, Tuple
import torch
from vllm import LLM
import wandb

from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH
from .utils import print_metrics, compute_retrieval_metrics, compute_answer_metrics, compute_all_answer_metrics
from .contriever import Retriever
from .query_generator.query_generator_sep import QueryGenerator
from .verifier import Reranker
from .answer_generator.answer_generator_sep import AnswerGenerator
from .stop_decider import StopDecider


def run_batch(
    retriever: Retriever,
    query_generator: QueryGenerator,
    reranker: Reranker,
    answer_generator: AnswerGenerator,
    stop_decider: StopDecider,
    questions: List[Dict[str, Any]],
    max_iterations: int = 5,
    max_search: int = 10,
    traces_path: str = None,
    stop_log_path: str = None,
    log_trace: bool = False
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], List[str], Dict[str, List[List[float]]]]:
    
    final_questions = []
    final_batch_history = []
    final_traces = []
    final_predictions = []

    batch_history = [[] for _ in range(len(questions))]
    traces = ["" for _ in questions]
    iter_count = 0

    stop_logs = []

    while questions:
        start_time = time.time()

        new_traces, search_queries = query_generator.batch_generate(questions, traces)
        traces = new_traces

        if log_trace:
            print(f"\n========== Iteration {iter_count + 1} ==========")
            for q, query, tr in zip(questions, search_queries, traces):
                print(f"[TRACE] Question: {q['question']}")
                print(f"|  Generated Query: {query}")
                print(f"|  Current Trace:\n{tr}\n")

        search_batch_size = 16
        batch_docs = []
        for i in range(0, len(search_queries), search_batch_size):
            batch_queries = search_queries[i:i + search_batch_size]
            batch_docs.extend(retriever.search(batch_queries, max_search))

        batch_scores = reranker.batch_rank(search_queries, batch_docs)

        intermediate_traces = []
        for idx, (question, history, trace, docs, scores) in enumerate(
            zip(questions, batch_history, traces, batch_docs, batch_scores)
        ):
            for i, doc in enumerate(docs):
                if doc["id"] in {d["id"] for d in history}:
                    scores[i] = float("-inf")
            
            selected_doc = docs[scores.argmax()]
            new_history = history + [selected_doc]
            
            new_trace = trace + f"\nDocument: {selected_doc['text']}"

            inter_traces, inter_answers = answer_generator.batch_generate_answers(
                [question], [new_trace], [False]
            )
            inter_trace = inter_traces[0]
            intermediate_traces.append((question, new_history, inter_trace, selected_doc, scores.max(), inter_answers[0]))

            if log_trace:
                print(f"[TRACE] QID={question['id']} - Selected Document")
                print(f"|  Score: {scores.max():.4f}")
                print(f"|  Document: {selected_doc['text'][:100]}...")
                print(f"|  Intermediate Answer: {inter_answers[0]}\n")

        next_questions = []
        next_batch_history = []
        next_traces = []

        stop_traces = [item[2] for item in intermediate_traces]
        stop_decisions = stop_decider.batch_decide(questions, stop_traces)

        for (question, new_history, inter_trace, selected_doc, score, inter_answer), decision in zip(
            intermediate_traces, stop_decisions
        ):
            if log_trace:
                print(f"[TRACE] QID={question['id']} - Stop Decision: {decision}")

            if decision == "STOP":
                final_traces_, final_answers = answer_generator.batch_generate_answers(
                    [question], [inter_trace], [True]
                )
                
                final_questions.append(question)
                final_batch_history.append(new_history)
                final_traces.append(final_traces_[0])
                final_predictions.append(final_answers[0])
                
                stop_logs.append({
                    "question_id": question["id"],
                    "gold_hop": len(question.get("question_decomposition", [])),
                    "stop_iter": iter_count + 1
                })
                
                if log_trace:
                    print(f"|  Final Answer: {final_answers[0]}\n")
            else:
                next_questions.append(question)
                next_batch_history.append(new_history)
                next_traces.append(inter_trace)

        questions = next_questions
        batch_history = next_batch_history
        traces = next_traces

        print(f"Iteration {iter_count+1} completed in {time.time() - start_time:.2f} seconds")
        print(f"Remaining questions: {len(questions)}")
        print(f"Completed questions: {len(final_questions)}\n")

        iter_count += 1
        if iter_count >= max_iterations:
            if questions:
                print(f"Max iterations reached. Generating final answers for {len(questions)} remaining questions.")
                
                final_traces_, final_answers = answer_generator.batch_generate_answers(
                    questions, traces, [True] * len(questions)
                )
                
                final_questions.extend(questions)
                final_batch_history.extend(batch_history)
                final_traces.extend(final_traces_)
                final_predictions.extend(final_answers)
                
                for question in questions:
                    stop_logs.append({
                        "question_id": question["id"],
                        "gold_hop": len(question.get("question_decomposition", [])),
                        "stop_iter": iter_count
                    })
            break

    if log_trace:
        print("\nFinal Questions and History:\n")
        for question, history in zip(final_questions, final_batch_history):
            print(f"1. Question: {question['question']}")
            print("2. History:")
            for doc in history:
                print(f"  Passage: {doc['text']}")
            print()

    em_list, precision_list, recall_list, f1_list = compute_retrieval_metrics(final_questions, final_batch_history, stop_logs)

    if stop_log_path:
        with open(stop_log_path, 'a', encoding='utf-8') as f:
            for log in stop_logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')

    ans_em_list, ans_f1_list = compute_answer_metrics(final_questions, final_predictions)
    
    if traces_path:
        all_ans_em_list, all_ans_f1_list = compute_all_answer_metrics(final_questions, final_predictions)
        with open(traces_path, 'a', encoding='utf-8') as f:
            for question, trace, prediction, em, f1 in zip(
                final_questions, final_traces, final_predictions, all_ans_em_list, all_ans_f1_list
            ):
                info = {
                    "question_id": question["id"],
                    "question": question.get("question", ""),
                    "answer": question.get("answer", ""),
                    "answer_aliases": question.get("answer_aliases", []),
                    "trace": trace,
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
            print(f"\nQID: {q['id']}")
            print(f"Question: {q['question']}")
            print(f"Gold Answer: {q.get('answer', 'N/A')}")
            print(f"Prediction: {pred}")
            print(f"Number of Documents Retrieved: {len(hist)}")
            print("-" * 80)

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
    query_generator_group.add_argument("--qg-max-gen-length", type=int, default=100, help="Maximum generation length for query generator")
    query_generator_group.add_argument("--qg-temperature", type=float, default=0.7, help="Temperature for query generator")
    query_generator_group.add_argument("--qg-top-p", type=float, default=0.9, help="Top-p sampling for query generator")

    reranker_group = parser.add_argument_group("Reranker Options")
    reranker_group.add_argument("--reranker-model-id", type=str, default="BAAI/bge-reranker-v2-m3", help="Model ID for reranker")
    reranker_group.add_argument("--reranker-batch-size", type=int, default=8, help="Batch size for reranker")
    reranker_group.add_argument("--reranker-max-length", type=int, default=DEBERTA_MAX_LENGTH, help="Maximum length for reranker input")

    answer_generator_group = parser.add_argument_group("Answer Generator Options")
    answer_generator_group.add_argument("--ag-max-gen-length", type=int, default=200, help="Maximum generation length for answer generator")
    answer_generator_group.add_argument("--ag-temperature", type=float, default=0.7, help="Temperature for answer generator")
    answer_generator_group.add_argument("--ag-top-p", type=float, default=0.9, help="Top-p sampling for answer generator")

    stop_decider_group = parser.add_argument_group("Stop Decider Options")
    stop_decider_group.add_argument("--sd-max-gen-length", type=int, default=50, help="Maximum generation length for stop decider")
    stop_decider_group.add_argument("--sd-temperature", type=float, default=0.1, help="Temperature for stop decider")
    stop_decider_group.add_argument("--sd-top-p", type=float, default=0.9, help="Top-p sampling for stop decider")

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--questions", type=str, required=True, help="Questions file path")
    main_group.add_argument("--batch-size", type=int, default=32, help="Batch size for processing questions")
    main_group.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")
    main_group.add_argument("--max-search", type=int, default=10, help="Maximum number of passages to retrieve")
    main_group.add_argument("--traces-path", type=str, help="Path to save traces")
    main_group.add_argument("--output-path", type=str, help="Path to save predictions and metrics")
    main_group.add_argument("--stop-log-path", type=str, default=None, help="Optional JSONL path; Path to the JSONL file where stopping logs are written")
    main_group.add_argument("--log-trace", action="store_true", help="Enable detailed trace logging")

    args = parser.parse_args()
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
        ###
        max_model_len=8192,
    )

    query_generator = QueryGenerator(
        llm=shared_llm,
        max_gen_length=args.qg_max_gen_length,
        temperature=args.qg_temperature,
        top_p=args.qg_top_p,
    )

    reranker = Reranker(
        model_id=args.reranker_model_id,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
    )

    answer_generator = AnswerGenerator(
        llm=shared_llm,
        max_gen_length=args.ag_max_gen_length,
        temperature=args.ag_temperature,
        top_p=args.ag_top_p,
    )

    stop_decider = StopDecider(
        llm=shared_llm,
        max_gen_length=args.sd_max_gen_length,
        temperature=args.sd_temperature,
        top_p=args.sd_top_p,
    )

    if args.traces_path:
        open(args.traces_path, "w", encoding="utf-8").close()

    if args.stop_log_path:
        open(args.stop_log_path, "w", encoding="utf-8").close()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = f.readlines()
        rd.shuffle(questions)
        questions = [json.loads(q) for q in questions]

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
            answer_generator=answer_generator,
            stop_decider=stop_decider,
            questions=batch_questions,
            max_iterations=args.max_iterations,
            max_search=args.max_search,
            traces_path=args.traces_path,
            stop_log_path=args.stop_log_path,
            log_trace=args.log_trace
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

        print("\n===== BATCH ANSWER METRICS =====")
        print_metrics(batch_metrics["answer"]["em"], "EM")
        print_metrics(batch_metrics["answer"]["f1"], "F1")

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