import argparse
import pandas as pd


def pick_values(threshold):
    def wrapper(subdf):
        over = subdf[subdf["score1"] - subdf["score2"] > threshold]
        chosen = over.iloc[0] if not over.empty else subdf.iloc[-1]
        return chosen[["em", "f1"]]
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

        result['num_hops'] = result['question_id'].str[0]

        print("\nDetailed summary:")
        summary = result.groupby('num_hops')[['em', 'f1']].agg(['mean', 'count'])
        print(summary)

        overall_averages = result[['em', 'f1']].mean()
        print("\nOverall averages:")
        print(overall_averages)
