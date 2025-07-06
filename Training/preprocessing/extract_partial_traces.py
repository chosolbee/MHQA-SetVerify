import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Extract partial traces from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
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
        answer = data["answer"]
        answer_aliases = data["answer_aliases"]
        answers = [answer] + answer_aliases
        trace = data["trace"]

        partial_trace = ""
        iter_cnt = 0
        for line in trace.split("\n"):
            partial_trace += line + "\n"
            if line.startswith("Intermediate answer: "):
                cnt += 1
                iter_cnt += 1
                with open(args.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "question_id": question_id,
                        "question": question,
                        "answers": answers,
                        "trace": partial_trace.strip(),
                        "iter_cnt": iter_cnt,
                    }, ensure_ascii=False) + "\n")

    print(f"Dataset Size: {cnt}")
    print("Extraction completed successfully.")
