import json
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os


def analyze_and_split_data(file_path, train_ratio=0.8, eval_ratio=0.1, random_state=42, output_dir=None, target_label='f1'):
    assert abs(train_ratio + eval_ratio - 1.0) < 1e-6, "The sum of ratios must equal 1."

    data_records = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                data_records.append(data)
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data_records)

    question_avg_label = df.groupby('question_id')[target_label].mean().reset_index()
    question_avg_label.columns = ['question_id', f'avg_{target_label}']

    train_questions, eval_questions = train_test_split(
        question_avg_label['question_id'],
        test_size=eval_ratio,
        stratify=pd.cut(question_avg_label[f'avg_{target_label}'], bins=5, labels=False),
        random_state=random_state
    )

    train_df = df[df['question_id'].isin(train_questions)]
    eval_df = df[df['question_id'].isin(eval_questions)]

    train_avg_label = train_df[target_label].mean()
    eval_avg_label = eval_df[target_label].mean()

    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%) - avg {target_label}: {train_avg_label:.6f}")
    print(f"Eval set: {len(eval_df)} samples ({len(eval_df)/len(df)*100:.1f}%) - avg {target_label}: {eval_avg_label:.6f}")

    output_files = save_splits(train_df, eval_df, file_path, output_dir)

    return output_files


def save_splits(train_df, eval_df, original_file_path, output_dir=None):
    filename = os.path.basename(original_file_path).replace('.jsonl', '')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_prefix = os.path.join(output_dir, filename)
    else:
        output_prefix = os.path.join(os.path.dirname(original_file_path), filename)

    splits = {
        'train': train_df,
        'eval': eval_df,
    }

    output_files = []

    for split_name, df in splits.items():
        output_file = f"{output_prefix}_{split_name}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                json.dump(row.to_dict(), f, ensure_ascii=False)
                f.write('\n')

        output_files.append(output_file)

    return output_files


def main():
    parser = argparse.ArgumentParser(description='Split JSONL data by question_id based on average prob')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Training data ratio (default: 0.9)')
    parser.add_argument('--eval-ratio', type=float, default=0.1, help='Evaluation data ratio (default: 0.1)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--target-label', type=str, default='f1', choices=['prob', 'em', 'f1'], help='Target label for splitting')

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: File '{args.input_path}' not found.")
        return

    if abs(args.train_ratio + args.eval_ratio - 1.0) >= 1e-6:
        print("Error: The sum of ratios must equal 1")
        return

    output_files = analyze_and_split_data(
        file_path=args.input_path,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        random_state=args.random_state,
        output_dir=args.output_dir,
        target_label=args.target_label,
    )

    print("Generated files:")
    for file in output_files:
        print(f"- {file}")


if __name__ == "__main__":
    main()
