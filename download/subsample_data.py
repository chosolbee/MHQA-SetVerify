import argparse
import json
import math
import random as rd
from config import DATASET_PATHS, DATASET_FIELDS


def parse_args():
    parser = argparse.ArgumentParser(description="Make Corpus from Dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"], help="Dataset name")
    parser.add_argument("--num-eval", type=int, default=1000, help="Number of evaluation samples")
    parser.add_argument("--num-test", type=int, default=1000, help="Number of test samples")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rd.seed(42)

    dataset_paths = DATASET_PATHS[args.dataset]

    with open(dataset_paths["dev"], "r", encoding="utf-8") as f:
        if args.dataset == "musique":
            data = [json.loads(line.strip()) for line in f]
        else:
            data = json.load(f)

    hop_counts = {}
    for item in data:
        num_hops = len(item[DATASET_FIELDS[args.dataset]["supporting_facts"]])
        hop_counts[num_hops] = hop_counts.get(num_hops, 0) + 1

    rd.shuffle(data)

    eval_data = []
    test_data = []

    for num_hops, count in hop_counts.items():
        eval_count = math.floor(count / len(data) * args.num_eval)
        test_count = math.floor(count / len(data) * args.num_test)

        hop_data = [item for item in data if len(item[DATASET_FIELDS[args.dataset]["supporting_facts"]]) == num_hops]
        eval_data.extend(hop_data[:eval_count])
        test_data.extend(hop_data[eval_count:eval_count + test_count])

    remaining_data = [item for item in data if item not in eval_data and item not in test_data]
    rd.shuffle(remaining_data)

    missing_eval = args.num_eval - len(eval_data)
    missing_test = args.num_test - len(test_data)
    eval_data.extend(remaining_data[:missing_eval])
    test_data.extend(remaining_data[missing_eval:missing_eval + missing_test])

    rd.shuffle(eval_data)
    rd.shuffle(test_data)

    with open(dataset_paths["eval_subsampled"], "w", encoding="utf-8") as f:
        if args.dataset == "musique":
            for item in eval_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(eval_data, f, ensure_ascii=False)

    with open(dataset_paths["test_subsampled"], "w", encoding="utf-8") as f:
        if args.dataset == "musique":
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(test_data, f, ensure_ascii=False)
