# app.py (diagnostic)
from retriever_with_classifier import LawRetrieverEnsemble
import traceback

def print_candidate(label, c):
    sec = c.get("section", "<no-section>")
    title = c.get("title", "<no-title>")
    cls = c.get("_cls_prob", 0.0)
    rr = c.get("_rerank", 0.0)
    final = c.get("_final_score", 0.0)
    print(f"{label}. {sec} | {title} | cls_prob={cls:.4f} | rerank={rr:.4f} | final={final:.4f}")

def main():
    retriever = LawRetrieverEnsemble(alpha=0.6, use_classifier=True)
    print("LawBot (ensemble) ready. Type a scenario (or 'quit'):")

    while True:
        try:
            q = input("\nScenario> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            break

        try:
            results = retriever.retrieve(q, top_k=6)
        except Exception as e:
            print("[ERROR] retrieve() raised an exception:")
            traceback.print_exc()
            continue

        if not results:
            print("[INFO] No candidates returned by retriever.retrieve(). Possible causes:")
            print("  - Chroma collection empty / not indexed (delete chroma_db and re-run indexing).")
            print("  - Query encoding failed.")
            print("  - top_k set too low (try top_k=10).")
            continue

        print("\nTop suggestions (section | title | cls_prob | rerank | final_score):")
        for i, r in enumerate(results, start=1):
            print_candidate(i, r)

        # Extra debug: print the classifier mapping (if present)
        if retriever.clf is not None:
            print("\n[DEBUG] Classifier available. Sample top predicted classes (probabilities):")
            try:
                cls_map = retriever.classifier_probs(q)
                if cls_map:
                    # show top 5 classifier predicted sections
                    items = sorted(cls_map.items(), key=lambda x: -x[1])[:6]
                    for sec, p in items:
                        print(f"  - {sec}: {p:.4f}")
                else:
                    print("  - classifier returned no mapping.")
            except Exception as e:
                print("  - classifier_probs() raised:", e)

if __name__ == "__main__":
    main()
