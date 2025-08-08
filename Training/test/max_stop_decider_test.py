import argparse
import pandas as pd

COLUMNS = ["em", "f1", "retrieval_em", "retrieval_precision", "retrieval_recall", "retrieval_f1"]

pd.set_option('display.max_columns', None)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute max test metrics from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")

    return parser.parse_args()


if __name__ == "__main__":
    print("Computing test metrics...")

    args = parse_args()

    df = pd.read_json(args.input_path, lines=True)

    df = df.sort_values(["question_id", "iter_cnt"])

    print("\n=== MAX ===")

    result = df.groupby('question_id').agg({**{column: "max" for column in COLUMNS}, "num_hops": "first"}).reset_index()

    print("\nDetailed summary (max):")
    summary = result.groupby('num_hops')[COLUMNS].agg(['mean', 'count'])
    print(summary)

    overall_averages = result[COLUMNS].mean()
    print("\nOverall averages (max):")
    print(overall_averages)

    print("\n=== NO STOP ===")

    result = df.groupby('question_id').agg({**{column: "last" for column in COLUMNS}, "num_hops": "first"}).reset_index()

    print("\nDetailed summary (no stop):")
    summary = result.groupby('num_hops')[COLUMNS].agg(['mean', 'count'])
    print(summary)

    overall_averages = result[COLUMNS].mean()
    print("\nOverall averages (no stop):")
    print(overall_averages)
