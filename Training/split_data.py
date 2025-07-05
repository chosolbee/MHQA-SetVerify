import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def analyze_and_split_data(file_path, train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1, random_state=42, output_dir=None):
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, "The sum of ratios must equal 1."
    
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
    
    question_avg_prob = df.groupby('question_id')['prob'].mean().reset_index()
    question_avg_prob.columns = ['question_id', 'avg_prob']
    
    train_questions, temp_questions = train_test_split(
        question_avg_prob['question_id'], 
        test_size=(eval_ratio + test_ratio),
        stratify=pd.cut(question_avg_prob['avg_prob'], bins=5, labels=False),
        random_state=random_state
    )
    
    eval_ratio_adjusted = eval_ratio / (eval_ratio + test_ratio)
    eval_questions, test_questions = train_test_split(
        temp_questions,
        test_size=(1 - eval_ratio_adjusted),
        random_state=random_state
    )
    
    train_df = df[df['question_id'].isin(train_questions)]
    eval_df = df[df['question_id'].isin(eval_questions)]
    test_df = df[df['question_id'].isin(test_questions)]
    
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Eval set: {len(eval_df)} samples ({len(eval_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    output_files = save_splits(train_df, eval_df, test_df, file_path, output_dir)
    
    return train_df, eval_df, test_df, output_files

def save_splits(train_df, eval_df, test_df, original_file_path, output_dir=None):
    filename = os.path.basename(original_file_path).replace('.jsonl', '')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_prefix = os.path.join(output_dir, filename)
    else:
        output_prefix = os.path.join(os.path.dirname(original_file_path), filename)
    
    splits = {
        'train': train_df,
        'eval': eval_df,
        'test': test_df
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
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio (default: 0.8)')
    parser.add_argument('--eval-ratio', type=float, default=0.1, help='Evaluation data ratio (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test data ratio (default: 0.1)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: File '{args.input_path}' not found.")
        return
    
    if abs(args.train_ratio + args.eval_ratio + args.test_ratio - 1.0) >= 1e-6:
        print("Error: The sum of train_ratio, eval_ratio, and test_ratio must equal 1.0")
        return
    
    train_df, eval_df, test_df, output_files = analyze_and_split_data(
        file_path=args.input_path,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        output_dir=args.output_dir
    )
    
    print("Generated files:")
    for file in output_files:
        print(f"- {file}")

if __name__ == "__main__":
    main()