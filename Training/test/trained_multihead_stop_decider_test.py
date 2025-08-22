import argparse
import pandas as pd

COLUMNS = ["em", "f1", "acc", "retrieval_em", "retrieval_precision", "retrieval_recall", "retrieval_f1", "iter_cnt"]

pd.set_option('display.max_columns', None)


def pick_values(threshold):
    def wrapper(subdf):
        over = subdf[subdf["score1"] - subdf["score2"] > threshold]
        chosen = over.iloc[0] if not over.empty else subdf.iloc[-1]
        return chosen[COLUMNS + ["num_hops"]]
    return wrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Compute answer scores from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--thresholds", type=float, nargs='+', required=True, help="List of thresholds to apply for score selection")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Reading data from {args.input_path}")

    df = pd.read_json(args.input_path, lines=True)
    df = df.sort_values(["question_id", "iter_cnt"])

    for threshold in args.thresholds:
        print(f"\nProcessing threshold: {threshold}")

        result = df.groupby("question_id", sort=False).apply(pick_values(threshold)).reset_index()

        print("\nDetailed summary:")
        summary = result.groupby('num_hops')[COLUMNS].agg(['mean', 'count'])
        print(summary)

        overall_averages = result[COLUMNS].mean()
        print("\nOverall averages:")
        print(overall_averages)
