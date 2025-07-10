import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compute max test metrics from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")

    return parser.parse_args()


if __name__ == "__main__":
    print("Computing max test metrics...")

    args = parse_args()

    df = pd.read_json(args.input_path, lines=True)

    df['hard_diff'] = (df['prob'] > df['max_cont_prob']).astype(int)

    result = df.groupby('question_id').agg({
        'em': 'max',
        'f1': 'max'
    }).reset_index()

    result['num_hops'] = result['question_id'].str[0]

    print("\nDetailed summary:")
    summary = result.groupby('num_hops')[['em', 'f1']].agg(['mean', 'count'])
    print(summary)

    overall_averages = result[['em', 'f1']].mean()
    print("\nOverall averages:")
    print(overall_averages)
