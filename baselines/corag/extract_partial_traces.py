import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Extract partial traces from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--max-iterations", type=int, default=8, help="Maximum number of iterations to extract traces")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Extracting partial traces...", flush=True)

    with open(args.input_path, "r", encoding="utf-8") as f:
        traces = [json.loads(line.strip()) for line in f]

    open(args.output_path, "w", encoding="utf-8").close()

    cnt = 0
    for data in traces:
        question_id = data["question_id"]
        question = data["question"]
        num_hops = data["gold_hop"]
        answers = [data["answer"]] + data["answer_aliases"]
        trace = data["trace"]
        history = data["history"]
        history_indices = data["history_indices"]

        partial_trace = ""
        iter_cnt = 0
        for line in trace.split("\n"):
            partial_trace += line + "\n"
            if line.startswith("Intermediate answer"):
                cnt += 1
                iter_cnt += 1
                partial_history = history[:history_indices[iter_cnt - 1]]
                partial_gold_cnt = sum(question_id + "-sf" in doc["id"] for doc in partial_history)
                with open(args.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "question_id": question_id,
                        "question": question,
                        "num_hops": num_hops,
                        "answers": answers,
                        "iter_cnt": iter_cnt,
                        "trace": partial_trace.strip(),
                        "history": partial_history,
                        "retrieval_em": int(partial_gold_cnt == num_hops and len(partial_history) == num_hops),
                        "retrieval_precision": partial_gold_cnt / len(partial_history),
                        "retrieval_recall": partial_gold_cnt / num_hops,
                        "retrieval_f1": (2 * partial_gold_cnt) / (len(partial_history) + num_hops),
                    }, ensure_ascii=False) + "\n")
            if iter_cnt >= args.max_iterations:
                break

    print(f"Dataset Size: {cnt}")
    print("Extraction completed successfully.")
