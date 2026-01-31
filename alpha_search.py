# alpha_search.py
import json
from retriever_with_classifier import LawRetrieverEnsemble
from evaluate_with_ensemble import evaluate
import numpy as np

def evaluate_alpha(alpha, retriever, test_cases, top_k=5):
    retriever.alpha = alpha
    res = evaluate(retriever, test_cases, top_k=top_k)
    # use overall aggregated score same as evaluate_with_ensemble
    overall = res["ndcg@k"]*0.4 + res["recall@k"]*0.3 + res["mrr"]*0.2 + res["precision@k"]*0.1
    return overall, res

if __name__ == "__main__":
    with open("data/test_cases_big.json","r",encoding="utf-8") as f:
        test_cases = json.load(f)
    retriever = LawRetrieverEnsemble(alpha=0.6, use_classifier=True)
    alphas = list(np.linspace(0.2, 0.9, 8))
    best = (None, -1, None)
    print("Testing alphas:", alphas)
    for a in alphas:
        print(f"\nEvaluating alpha={a:.2f} ...")
        overall, res = evaluate_alpha(a, retriever, test_cases, top_k=5)
        print(f" alpha={a:.2f} -> overall={overall:.4f}")
        if overall > best[1]:
            best = (a, overall, res)
    print("\nBEST alpha:", best[0], "overall:", best[1])
    print("Best metrics:")
    for k,v in best[2].items():
        print(f" {k}: {v}")
