import re
from collections import Counter
from typing import List, Dict, Any, Tuple


def print_metrics(metrics_list, metric_name="Metrics"):
    if not metrics_list or not any(metrics_list):
        print(f"No {metric_name} available.")
        return

    metrics_flat = [item for sublist in metrics_list for item in sublist]

    print(f"===== {metric_name} =====")
    print(f"Count: {[len(m) for m in metrics_list]}")

    averages = [sum(m) / len(m) if m else 0 for m in metrics_list]
    print(f"{metric_name} by hop: {[f'{avg:.4f}' for avg in averages]}")

    total_avg = sum(metrics_flat) / len(metrics_flat) if metrics_flat else 0
    print(f"Total {metric_name}: {total_avg:.4f}")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\b(a|an|the)\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0

    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def compute_retrieval_metrics(questions: List[Dict[str, Any]],
                              batch_history: List[List[Dict[str, Any]]],
                              stop_logs: List[Dict[str, Any]]) -> Tuple[List[List[float]], ...]:
    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    for question, history in zip(questions, batch_history):
        qid = question["id"]
        gold_hop = len(question.get("question_decomposition", []))
        correct = sum(int(qid + "-sf" in doc["id"]) for doc in history)
        retrieved = len(history)

        em = int(correct == gold_hop and retrieved == gold_hop)
        precision = correct / retrieved if retrieved else 0.0
        recall = correct / gold_hop if gold_hop else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        for log in stop_logs:
            if log["question_id"] == qid:
                log.update({"em": em, "precision": precision, "recall": recall, "f1": f1})
                break

        idx = min(max(gold_hop - 2, 0), 2)
        em_list[idx].append(em)
        precision_list[idx].append(precision)
        recall_list[idx].append(recall)
        f1_list[idx].append(f1)

    return em_list, precision_list, recall_list, f1_list


def compute_answer_metrics(questions: List[Dict[str, Any]],
                           predictions: List[str]) -> Tuple[List[List[float]], List[List[float]]]:
    em_list = [[], [], []]
    f1_list = [[], [], []]

    for question, prediction in zip(questions, predictions):
        hop = len(question.get("question_decomposition", []))
        idx = min(max(hop - 2, 0), 2)

        gold_answers = [question["answer"]] + question.get("answer_aliases", [])
        gold_answers = [normalize(g) for g in gold_answers]

        norm_pred = normalize(prediction)

        em = int(norm_pred in gold_answers)
        f1 = max(token_f1(norm_pred, g) for g in gold_answers)

        em_list[idx].append(em)
        f1_list[idx].append(f1)

    return em_list, f1_list


def compute_all_answer_metrics(questions: List[Dict[str, Any]],
                                 predictions: List[str]) -> Tuple[List[float], List[float]]:
    em_list = []
    f1_list = []

    for question, prediction in zip(questions, predictions):
        gold_answers = [question["answer"]] + question.get("answer_aliases", [])
        gold_answers = [normalize(g) for g in gold_answers]

        norm_pred = normalize(prediction)

        em = int(norm_pred in gold_answers)
        f1 = max(token_f1(norm_pred, g) for g in gold_answers)

        em_list.append(em)
        f1_list.append(f1)

    return em_list, f1_list
