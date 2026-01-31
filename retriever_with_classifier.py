# retriever_with_classifier.py
import os
import json
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import joblib

CLASSIFIER_PATH = "models/query_classifier.joblib"

class LawRetrieverEnsemble:
    def __init__(self, data_path="data/laws.json", chroma_db_path="./chroma_db", use_classifier=True, alpha=0.6):
        load_dotenv()
        with open(data_path, "r", encoding="utf-8") as f:
            self.laws = json.load(f)
        self.sections = [l["section"] for l in self.laws]

        print("Loading MPNet embedding model...")
        self.embed_model = SentenceTransformer("all-mpnet-base-v2")

        print("Initializing ChromaDB...")
        try:
            self.client = chromadb.PersistentClient(path=chroma_db_path)
        except Exception:
            self.client = chromadb.Client()
        try:
            self.collection = self.client.get_or_create_collection("indian_law_sections")
        except Exception:
            self.collection = self.client.create_collection("indian_law_sections")

        try:
            print("Loading cross-encoder...")
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print("[warning] cross-encoder load failed:", e)
            self.cross_encoder = None

        self.use_classifier = use_classifier
        self.alpha = alpha
        self.clf = None
        self.le = None
        self.clf_label_map = None  # maps classifier-internal label indices -> section name

        if use_classifier and os.path.exists(CLASSIFIER_PATH):
            ck = joblib.load(CLASSIFIER_PATH)
            self.clf = ck["clf"]
            self.le = ck["le"]
            # IMPORTANT FIX:
            # clf.classes_ contains the *encoded* integer labels that the classifier saw during training.
            # We must map those encoded labels back to the original section strings using the label encoder.
            try:
                encoded_class_indices = list(self.clf.classes_)
                # In many workflows, clf.classes_ are integers (the encoded labels). Use le.inverse_transform on them.
                self.clf_label_map = {idx: str(self.le.inverse_transform([idx])[0]) for idx in encoded_class_indices}
                print("Loaded query classifier. Classifier supports sections:", list(self.clf_label_map.values()))
            except Exception as e:
                # Fallback: if clf.classes_ are already strings, map directly
                print("[warning] could not inverse_transform classifier classes:", e)
                try:
                    self.clf_label_map = {c: c for c in self.clf.classes_}
                except Exception:
                    self.clf_label_map = None
        else:
            if use_classifier:
                print("Classifier not found; run train_query_classifier.py first.")

    def encode(self, texts):
        return self.embed_model.encode(texts, show_progress_bar=False)

    def base_retrieve(self, query, top_k=10):
        q_emb = self.embed_model.encode(query).tolist()
        try:
            res = self.collection.query(query_embeddings=[q_emb], n_results=top_k, include=["metadatas", "distances", "ids"])
        except Exception:
            res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0] if "distances" in res else []
        ids = res.get("ids", [[]])[0] if "ids" in res else []
        results = []
        for i, md in enumerate(metadatas):
            entry = dict(md) if isinstance(md, dict) else {"text": str(md)}
            if i < len(ids): entry["_id"] = ids[i]
            if i < len(distances): entry["_distance"] = float(distances[i])
            results.append(entry)
        return results

    def crossencode_scores(self, query, candidates: List[Dict]):
        if not self.cross_encoder or not candidates:
            return [0.0] * len(candidates)
        pair_inputs = []
        for c in candidates:
            text = (c.get("title","") + ". " + c.get("text","")).strip()
            pair_inputs.append((query, text))
        scores = self.cross_encoder.predict(pair_inputs)
        min_s, max_s = float(min(scores)), float(max(scores))
        if max_s - min_s < 1e-6:
            norm = [0.5 for _ in scores]
        else:
            norm = [(s - min_s) / (max_s - min_s) for s in scores]
        return norm

    def classifier_probs(self, query):
        """
        Return a mapping: section_string -> probability, but only for sections the classifier knows.
        Correctly handle mapping from classifier internal classes -> label encoder strings.
        """
        if not self.clf or not self.le or not self.clf_label_map:
            return None

        emb = self.embed_model.encode([query])
        probs = self.clf.predict_proba(emb)[0]  # probs array aligned with clf.classes_
        # Build mapping: section -> prob
        mapping = {}
        for idx_pos, encoded_label in enumerate(self.clf.classes_):
            # encoded_label is typically an integer corresponding to label_encoder.transform(...)
            try:
                section_name = self.clf_label_map[encoded_label]
            except Exception:
                # fallback: try inverse_transform with encoded_label
                try:
                    section_name = str(self.le.inverse_transform([encoded_label])[0])
                except Exception:
                    section_name = str(encoded_label)
            mapping[section_name] = float(probs[idx_pos])
        return mapping

    def retrieve(self, query, top_k=6, alpha=None):
        if alpha is None:
            alpha = self.alpha
        candidates = self.base_retrieve(query, top_k=top_k)
        rerank_norm = self.crossencode_scores(query, candidates) if self.cross_encoder else [0.0]*len(candidates)
        cls_map = self.classifier_probs(query) if self.use_classifier else None

        final = []
        for c, rscore in zip(candidates, rerank_norm):
            sec = c.get("section")
            cls_prob = cls_map.get(sec, 0.0) if cls_map else 0.0
            final_score = alpha * cls_prob + (1.0 - alpha) * rscore
            item = dict(c)
            item["_cls_prob"] = cls_prob
            item["_rerank"] = rscore
            item["_final_score"] = final_score
            final.append(item)

        final.sort(key=lambda x: -x["_final_score"])
        return final

if __name__ == "__main__":
    r = LawRetrieverEnsemble()
    q = "A car driver sped through a red light and caused injuries to pedestrians."
    res = r.retrieve(q, top_k=6)
    for r in res:
        print(r.get("section"), r.get("title"), "cls:", r["_cls_prob"], "rerank:", r["_rerank"], "final:", r["_final_score"])
