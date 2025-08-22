import argparse
from tqdm import tqdm
import pandas as pd

tqdm.pandas()


def parse_args():
    parser = argparse.ArgumentParser(description="Compute answer scores from the dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--target-label", type=str, default="prob", choices=["prob", "em", "f1", "acc"], help="Target label for training")
    parser.add_argument("--max-iterations", type=int, default=8, help="Maximum number of iterations to consider for each question")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_json(args.input_path, lines=True)
    df["id"] = df.index
    df["lookup_key"] = df["question_id"].astype(str) + "_iter" + df["iter_cnt"].astype(str)
    df_pivot = df.set_index("lookup_key")

    def process_trace(row):
        if row["iter_cnt"] == args.max_iterations:
            return row

        question_id = row["question_id"]
        iter_cnt = row["iter_cnt"]

        cont_keys = [f"{question_id}_iter{i}" for i in range(iter_cnt + 1, args.max_iterations)]

        cont_labels = []
        cont_ids = []

        for key in cont_keys:
            if key in df_pivot.index:
                cont_labels.append(df_pivot.loc[key, args.target_label])
                cont_ids.append(df_pivot.loc[key, "id"])

        cont_labels += [-1.0] * (args.max_iterations - 2 - len(cont_labels))
        cont_ids += [-1] * (args.max_iterations - 2 - len(cont_ids))
        cont_mask = [True] * (args.max_iterations - 1 - iter_cnt) + [False] * (iter_cnt - 1)

        last_key = f"{question_id}_iter{args.max_iterations}"
        last_label = df_pivot.loc[last_key, args.target_label] if last_key in df_pivot.index else None

        row[f"cont_{args.target_label}"] = cont_labels
        row["cont_ids"] = cont_ids
        row["cont_mask"] = cont_mask
        row[f"last_{args.target_label}"] = last_label

        return row

    df = df.progress_apply(process_trace, axis=1)
    df = df.drop(columns=['lookup_key'])
    df = df.sort_values(["question_id", "iter_cnt"]).reset_index(drop=True)

    df.to_json(args.output_path, orient="records", lines=True, force_ascii=False, mode="w")
