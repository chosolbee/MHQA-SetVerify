import argparse
import json
import random as rd
from config import DATASET_PATHS


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

    rd.shuffle(data)
    eval_data = data[:args.num_eval]
    test_data = data[args.num_eval:args.num_eval + args.num_test]

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
