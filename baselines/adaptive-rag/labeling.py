import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict


def load_results(filepath: str) -> Dict[str, Dict]:
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            results[data['question_id']] = data
    return results


def determine_label(no_result: Dict, single_result: Dict) -> str:

    if no_result.get('acc', 0) == 1:
        return 'A'
    
    if single_result.get('acc', 0) == 1:
        return 'B'
    
    return 'C'


def silver_labeling(no_results: Dict, single_results: Dict) -> Dict[str, str]:

    labels = {}
    
    all_qids = set(no_results.keys()) & set(single_results.keys())
    
    for qid in all_qids:
        no_result = no_results[qid]
        single_result = single_results[qid]
        
        label = determine_label(no_result, single_result)
        labels[qid] = label
    
    return labels


def create_labeled_dataset(labels: Dict[str, str], original_results: Dict, output_path: str):

    labeled_data = []
    
    for qid, label in labels.items():
        if qid in original_results:
            result = original_results[qid]
            labeled_entry = {
                'question_id': qid,
                'question': result.get('question', ''),
                'answer': result.get('answer', ''),
                'gold_hop': result.get('gold_hop', 0),
                'label': label
            }
            labeled_data.append(labeled_entry)
    
    labeled_data.sort(key=lambda x: x['question_id'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in labeled_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def analyze_labeling_distribution(labels: Dict[str, str]) -> Dict[str, int]:
    distribution = defaultdict(int)
    for label in labels.values():
        distribution[label] += 1
    return dict(distribution)


def print_statistics(labels: Dict[str, str], no_results: Dict, single_results: Dict):
    total = len(labels)
    distribution = analyze_labeling_distribution(labels)
    
    print(f"\n===== LABELING STATISTICS =====")
    print(f"Total questions: {total}")
    print(f"Label A (no-retrieval): {distribution.get('A', 0)} ({distribution.get('A', 0)/total*100:.1f}%)")
    print(f"Label B (single-retrieval): {distribution.get('B', 0)} ({distribution.get('B', 0)/total*100:.1f}%)")
    print(f"Label C (multi-retrieval): {distribution.get('C', 0)} ({distribution.get('C', 0)/total*100:.1f}%)")
    
    no_correct = sum(1 for qid in labels.keys() if no_results[qid].get('acc', 0) == 1)
    single_correct = sum(1 for qid in labels.keys() if single_results[qid].get('acc', 0) == 1)
    
    print(f"\n===== STRATEGY ACCURACIES =====")
    print(f"No-retrieval accuracy: {no_correct}/{total} ({no_correct/total*100:.1f}%)")
    print(f"Single-retrieval accuracy: {single_correct}/{total} ({single_correct/total*100:.1f}%)")
    
    no_correct_qids = {qid for qid in labels.keys() if no_results[qid].get('acc', 0) == 1}
    single_correct_qids = {qid for qid in labels.keys() if single_results[qid].get('acc', 0) == 1}
    
    overlap = len(no_correct_qids & single_correct_qids)
    print(f"\nQuestions correct by both no and single: {overlap}")
    print(f"Questions correct by single but not no: {len(single_correct_qids - no_correct_qids)}")
    print(f"Questions correct by no but not single: {len(no_correct_qids - single_correct_qids)}")


def validate_inputs(no_results: Dict, single_results: Dict):
    no_qids = set(no_results.keys())
    single_qids = set(single_results.keys())
    
    if no_qids != single_qids:
        missing_in_single = no_qids - single_qids
        missing_in_no = single_qids - no_qids
        
        print("WARNING: Question ID mismatch between no and single retrieval results!")
        if missing_in_single:
            print(f"Missing in single: {len(missing_in_single)} questions")
        if missing_in_no:
            print(f"Missing in no: {len(missing_in_no)} questions")
    
    return no_qids & single_qids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label questions based on retrieval strategy performance"
    )
    
    parser.add_argument(
        "--no-retrieval-results", 
        type=str, 
        required=True,
        help="Path to no-retrieval results JSONL file"
    )
    
    parser.add_argument(
        "--single-retrieval-results", 
        type=str, 
        required=True,
        help="Path to single-retrieval results JSONL file"
    )
    
    parser.add_argument(
        "--output-path", 
        type=str, 
        required=True,
        help="Path to save the labeled dataset"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed statistics"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Loading results...")
    no_results = load_results(args.no_retrieval_results)
    single_results = load_results(args.single_retrieval_results)
    
    print(f"Loaded {len(no_results)} no-retrieval results")
    print(f"Loaded {len(single_results)} single-retrieval results")
    
    valid_qids = validate_inputs(no_results, single_results)
    print(f"Valid questions for labeling: {len(valid_qids)}")
    
    print("\nPerforming silver labeling...")
    labels = silver_labeling(no_results, single_results)
    
    print(f"Creating labeled dataset at {args.output_path}...")
    create_labeled_dataset(labels, no_results, args.output_path)
    
    if args.verbose:
        print_statistics(labels, no_results, single_results)
    else:
        distribution = analyze_labeling_distribution(labels)
        print(f"\nLabeling complete! Distribution: A={distribution.get('A', 0)}, B={distribution.get('B', 0)}, C={distribution.get('C', 0)}")
    
    print(f"Labeled dataset saved to: {args.output_path}")


if __name__ == "__main__":
    main()