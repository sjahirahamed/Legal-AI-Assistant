# evaluate_with_ensemble.py
import json
import time
import math
from retriever_with_classifier import LawRetrieverEnsemble

def precision_at_k(predicted, expected, k):
    pred_k = predicted[:k]
    return len([p for p in pred_k if p in expected]) / k

def recall_at_k(predicted, expected, k):
    if not expected:
        return 0.0
    pred_k = predicted[:k]
    return len([p for p in pred_k if p in expected]) / len(expected)

def reciprocal_rank(predicted, expected):
    for i, p in enumerate(predicted, start=1):
        if p in expected:
            return 1.0 / i
    return 0.0

def dcg_at_k(predicted, expected, k):
    dcg = 0.0
    for i, p in enumerate(predicted[:k], start=1):
        rel = 1.0 if p in expected else 0.0
        dcg += (2**rel - 1.0) / math.log2(i + 1)
    return dcg

def idcg_at_k(expected, k):
    ideal_rels = [1.0] * min(len(expected), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2**rel - 1.0) / math.log2(i + 1)
    return idcg

def ndcg_at_k(predicted, expected, k):
    idcg = idcg_at_k(expected, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(predicted, expected, k) / idcg

def evaluate(retriever, test_cases, top_k=5):
    metrics = {"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0, "avg_latency_s": 0.0}
    n = len(test_cases)
    total_time = 0.0
    for case in test_cases:
        q = case["query"]
        expected = case.get("expected_sections", [])
        t0 = time.time()
        res = retriever.retrieve(q, top_k=top_k)
        t1 = time.time()
        total_time += (t1 - t0)
        predicted = [r.get("section") for r in res]
        metrics["precision@k"] += precision_at_k(predicted, expected, top_k)
        metrics["recall@k"] += recall_at_k(predicted, expected, top_k)
        metrics["mrr"] += reciprocal_rank(predicted, expected)
        metrics["ndcg@k"] += ndcg_at_k(predicted, expected, top_k)

    for k in ["precision@k", "recall@k", "mrr", "ndcg@k"]:
        metrics[k] /= max(1, n)
    metrics["avg_latency_s"] = total_time / max(1, n)
    try:
        metrics["indexed_count"] = retriever.collection.count()
    except Exception:
        metrics["indexed_count"] = "unknown"
    return metrics

if __name__ == "__main__":
    retriever = LawRetrieverEnsemble(alpha=0.6, use_classifier=True)
    with open("data/test_cases_big.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases.")
    res = evaluate(retriever, test_cases, top_k=5)
    print("=== Evaluation ===")
    for k, v in res.items():
        print(f"{k}: {v}")
    # Compute aggregated accuracy (weights same as before)
    w_ndcg = 0.4
    w_recall = 0.3
    w_mrr = 0.2
    w_prec = 0.1
    overall = res["ndcg@k"]*w_ndcg + res["recall@k"]*w_recall + res["mrr"]*w_mrr + res["precision@k"]*w_prec
    print(f"\nOverall aggregated score: {overall:.4f} (~{overall*100:.1f}%)")
