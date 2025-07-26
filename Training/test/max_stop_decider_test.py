import argparse
import pandas as pd


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

    result = df.groupby('question_id').agg({
        'em': 'max',
        'f1': 'max'
    }).reset_index()

    result['num_hops'] = result['question_id'].str[0]

    print("\nDetailed summary (max):")
    summary = result.groupby('num_hops')[['em', 'f1']].agg(['mean', 'count'])
    print(summary)

    overall_averages = result[['em', 'f1']].mean()
    print("\nOverall averages (max):")
    print(overall_averages)

    print("\n=== NO STOP ===")

    result = df.groupby('question_id').agg({
        'em': 'last',
        'f1': 'last'
    }).reset_index()

    result['num_hops'] = result['question_id'].str[0]

    print("\nDetailed summary (no stop):")
    summary = result.groupby('num_hops')[['em', 'f1']].agg(['mean', 'count'])
    print(summary)

    overall_averages = result[['em', 'f1']].mean()
    print("\nOverall averages (no stop):")
    print(overall_averages)
