# evaluate_retriever.py
import json
import time
import math
from retriever import LawRetriever

def precision_at_k(predicted: list, expected: list, k: int):
    pred_k = predicted[:k]
    return len([p for p in pred_k if p in expected]) / k

def recall_at_k(predicted: list, expected: list, k: int):
    pred_k = predicted[:k]
    if len(expected) == 0:
        return 0.0
    return len([p for p in pred_k if p in expected]) / len(expected)

def reciprocal_rank(predicted: list, expected: list):
    for i, p in enumerate(predicted, start=1):
        if p in expected:
            return 1.0 / i
    return 0.0

def dcg_at_k(predicted: list, expected: list, k: int):
    dcg = 0.0
    for i, p in enumerate(predicted[:k], start=1):
        rel = 1.0 if p in expected else 0.0
        dcg += (2**rel - 1.0) / math.log2(i + 1)
    return dcg

def idcg_at_k(expected: list, k: int):
    ideal_rels = [1.0] * min(len(expected), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2**rel - 1.0) / math.log2(i + 1)
    return idcg

def ndcg_at_k(predicted: list, expected: list, k: int):
    idcg = idcg_at_k(expected, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(predicted, expected, k) / idcg

def evaluate(retriever: LawRetriever, test_cases: list, top_k=5):
    n = len(test_cases)
    metrics = {"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0, "avg_latency_s": 0.0}
    total_latency = 0.0

    for case in test_cases:
        q = case["query"]
        expected = case.get("expected_sections", [])
        t0 = time.time()
        matches = retriever.find_relevant_laws(q, top_k=top_k)
        t1 = time.time()
        latency = t1 - t0
        total_latency += latency

        # predicted ids
        predicted = [m.get("section") for m in matches]

        metrics["precision@k"] += precision_at_k(predicted, expected, top_k)
        metrics["recall@k"] += recall_at_k(predicted, expected, top_k)
        metrics["mrr"] += reciprocal_rank(predicted, expected)
        metrics["ndcg@k"] += ndcg_at_k(predicted, expected, top_k)

    # average
    for k in ["precision@k", "recall@k", "mrr", "ndcg@k"]:
        metrics[k] /= max(1, n)
    metrics["avg_latency_s"] = total_latency / max(1, n)
    # index count
    try:
        metrics["indexed_count"] = retriever.collection.count()
    except Exception:
        metrics["indexed_count"] = "unknown"
    return metrics

if __name__ == "__main__":
    retriever = LawRetriever(data_path="data/laws.json")
    # Load or create test_cases.json
    import os
    tc_path = "data/test_cases.json"
    if not os.path.exists(tc_path):
        # create a small starter set
        starter = [
            {
                "query": "A person stabbed someone with a knife during a fight; the victim survived with serious injuries.",
                "expected_sections": ["IPC 324", "IPC 325"]
            },
            {
                "query": "Someone drove their car rashly and hit a pedestrian causing death.",
                "expected_sections": ["IPC 304A"]
            },
            {
                "query": "A man forcibly had sexual intercourse with a woman without her consent.",
                "expected_sections": ["IPC 375", "IPC 376"]
            },
            {
                "query": "A domestic husband continually mentally harasses and threatens his wife.",
                "expected_sections": ["IPC 498A", "IPC 506"]
            },
            {
                "query": "A shop assistant steals cash from the shop's cash register.",
                "expected_sections": ["IPC 381", "IPC 379"]
            }
        ]
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(starter, f, indent=2)
        test_cases = starter
    else:
        with open(tc_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases.")
    res = evaluate(retriever, test_cases, top_k=5)
    print("=== Evaluation ===")
    for k, v in res.items():
        print(f"{k}: {v}")
