import argparse
from tqdm import tqdm
import pandas as pd

tqdm.pandas()


def compute_max_cont_prob(group):
    max_cont_probs = []
    for _, row in group.iterrows():
        future_mask = group["iter_cnt"] > row["iter_cnt"]
        if future_mask.any():
            max_cont_prob = group.loc[future_mask, "prob"].max()
        else:
            max_cont_prob = 0.0
        max_cont_probs.append(max_cont_prob)

    group = group.copy()
    group['max_cont_prob'] = max_cont_probs
    return group


def parse_args():
    parser = argparse.ArgumentParser(description="Extract partial traces from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading data...")

    df = pd.read_json(args.input_path, lines=True)

    print(f"Loaded {len(df)} entries")
    print("Computing max_cont_prob...")

    df = df.sort_values(["question_id", "iter_cnt"])
    df = df.groupby("question_id", group_keys=False).progress_apply(compute_max_cont_prob)

    print("Writing output...")

    df.to_json(args.output_path, orient="records", lines=True)

    print(f"Processing complete. Output written to {args.output_path}")
