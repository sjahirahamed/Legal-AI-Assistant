# reindex_chroma.py
"""
Safe script to (re)create the Chroma collection "indian_law_sections"
and index all laws from data/laws.json using MPNet embeddings.
Run: python reindex_chroma.py
"""

import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

DATA_PATH = Path("data/laws.json")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "indian_law_sections"

def load_laws(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        laws = json.load(f)
    # ensure each law has required fields
    cleaned = []
    for law in laws:
        sec = law.get("section") or law.get("id") or None
        title = law.get("title", "")
        text = law.get("text", "")
        if not sec:
            # skip malformed entry
            continue
        cleaned.append({"section": sec, "title": title, "text": text})
    return cleaned

def main():
    if not DATA_PATH.exists():
        print("ERROR: data/laws.json not found. Please add the file and try again.")
        return

    laws = load_laws(DATA_PATH)
    print(f"Loaded {len(laws)} laws from {DATA_PATH}")

    print("Loading embedding model (all-mpnet-base-v2)...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("Connecting to ChromaDB (persistent)...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
    except Exception:
        client = chromadb.Client()

    # delete existing collection if exists (best-effort)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection (if any).")
    except Exception:
        # some chroma versions don't support delete_collection
        pass

    # create collection
    try:
        collection = client.get_or_create_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(COLLECTION_NAME)

    ids = []
    metadatas = []
    documents = []
    embeddings = []

    for law in laws:
        doc_text = (law.get("title", "") + ". " + law.get("text", "")).strip()
        emb = model.encode(doc_text)
        ids.append(law["section"])
        metadatas.append(law)
        documents.append(doc_text)
        embeddings.append(emb.tolist())

    # add in batches to avoid memory issues
    batch = 32
    for i in range(0, len(ids), batch):
        batch_ids = ids[i:i+batch]
        batch_docs = documents[i:i+batch]
        batch_embs = embeddings[i:i+batch]
        batch_meta = metadatas[i:i+batch]
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=batch_meta
            )
        except TypeError:
            # older chroma versions may expect different parameter order/names
            collection.add(
                ids=batch_ids,
                embeddings=batch_embs,
                metadatas=batch_meta,
                documents=batch_docs
            )
    try:
        cnt = collection.count()
    except Exception:
        try:
            info = collection.get()
            cnt = len(info.get("ids", [])) if info else 0
        except Exception:
            cnt = "unknown"

    print(f"Indexed {len(ids)} laws into ChromaDB. Collection count (reported): {cnt}")
    print("Done. Now run: python app.py  (or python debug_retriever.py)")

if __name__ == "__main__":
    main()
