# debug_retriever.py
import json, traceback
from retriever_with_classifier import LawRetrieverEnsemble

def safe_print(title, obj):
    print("\n" + "="*8 + f" {title} " + "="*8)
    try:
        print(obj)
    except Exception as e:
        print("ERROR PRINTING OBJECT:", e)

def main():
    try:
        r = LawRetrieverEnsemble(alpha=0.6, use_classifier=True)
    except Exception as e:
        print("Failed to initialize retriever:", e)
        traceback.print_exc()
        return

    # 1) Collection count
    try:
        cnt = r.collection.count()
    except Exception as e:
        print("collection.count() raised:", e)
        try:
            info = r.collection.get()
            cnt = len(info.get("ids", []))
            print("Got count from get():", cnt)
        except Exception as e2:
            print("collection.get() also failed:", e2)
            cnt = None
    print("Chroma collection count:", cnt)

    # 2) Dump small sample from collection.get() (first 5 ids / metadatas)
    try:
        info = r.collection.get()
        # info shape depends on chroma version
        safe_print("collection.get() raw", info)
        ids = info.get("ids", []) if isinstance(info, dict) else []
        metas = info.get("metadatas", []) if isinstance(info, dict) else []
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if metas and isinstance(metas[0], list):
            metas = metas[0]
        print("Sample ids (first 10):", ids[:10])
        print("Sample metadatas (first 3):", metas[:3])
    except Exception as e:
        print("collection.get() raised:", e)

    # 3) Show embedding for a debug query
    query = "A man robbed a woman by snatching her purse on a deserted street"
    try:
        emb = r.embed_model.encode(query)
        print("Query embedding shape:", getattr(emb, "shape", len(emb)))
    except Exception as e:
        print("Embedding encode failed:", e)
        emb = None

    # 4) Try a raw collection.query with include
    try:
        print("\nAttempting collection.query(query_embeddings=[emb], n_results=6, include=['metadatas','ids','distances'])")
        if emb is not None:
            res = r.collection.query(query_embeddings=[emb.tolist()], n_results=6, include=["metadatas","ids","distances"])
            safe_print("Raw query (include) response", res)
        else:
            print("Skipping include query: emb is None")
    except Exception as e:
        print("collection.query(include=...) raised:", e)
        try:
            res = r.collection.query(query_embeddings=[emb.tolist()], n_results=6)
            safe_print("Raw query (no-include) response", res)
        except Exception as e2:
            print("collection.query(no-include) also raised:", e2)

    # 5) Use retriever.retrieve() (full pipeline)
    try:
        print("\nCalling retriever.retrieve() (top_k=6)")
        results = r.retrieve(query, top_k=6)
        safe_print("retrieve() results", results)
    except Exception as e:
        print("retriever.retrieve() raised:", e)
        traceback.print_exc()

    # 6) If count is 0 or None, auto-index (safe)
    if not cnt:
        print("\nIndex appears empty. Running r.index_laws() to (re)index now...")
        try:
            r.index_laws()
            print("Indexing done. Please re-run this script (or app.py).")
        except Exception as e:
            print("index_laws() failed:", e)
            traceback.print_exc()

if __name__ == "__main__":
    main()
