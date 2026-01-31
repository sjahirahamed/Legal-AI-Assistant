# retriever.py
import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

import chromadb

# Optional Gemini import
try:
    from google import genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# Simple synonyms map for query expansion (extend as needed)
SYNONYMS = {
    "hurt": ["injury", "wound", "harm"],
    "injure": ["hurt", "wound"],
    "kill": ["murder", "slay"],
    "steal": ["theft", "rob", "robbery", "snatch"],
    "threaten": ["intimidate", "menace"],
    "cheat": ["fraud", "deceive"],
    "burn": ["set fire", "arson"],
    "hit": ["strike", "knock", "collide"],
    "drive": ["drive", "rash driving", "negligent driving"]
}

def expand_query_simple(query: str, synonyms_map: Dict[str, List[str]], max_terms:int=5) -> str:
    """
    Expand query by appending synonyms found in the map for tokens present in query.
    This is a light-weight synonym expansion to improve recall.
    """
    q = query.lower()
    additions = []
    for word, syns in synonyms_map.items():
        if word in q:
            for s in syns:
                if s not in q and len(additions) < max_terms:
                    additions.append(s)
    if additions:
        expanded = query + " | " + " ".join(additions)
        return expanded
    return query

class LawRetriever:
    def __init__(
        self,
        data_path: str = "data/laws.json",
        chroma_db_path: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        force_reindex: bool = False,
        use_local_reranker: bool = True
    ):
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", None)
        if self.gemini_api_key and not HAS_GEMINI:
            print("[warning] GEMINI key present but genai client not installed; Gemini features disabled until installed.")

        # Load laws
        with open(data_path, "r", encoding="utf-8") as f:
            self.laws: List[Dict] = json.load(f)

        # Embedding model (better quality)
        print("Loading embedding model (mpnet)...")
        # note: use the model id in SentenceTransformer constructor
        self.model = SentenceTransformer("all-mpnet-base-v2")

        # Cross-encoder reranker (local)
        self.use_local_reranker = use_local_reranker
        self.cross_encoder = None
        if use_local_reranker:
            try:
                print("Loading cross-encoder reranker...")
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                print("[warning] Could not load cross-encoder:", e)
                self.cross_encoder = None

        # Init ChromaDB
        print("Initializing ChromaDB...")
        try:
            self.client = chromadb.PersistentClient(path=chroma_db_path)
        except Exception:
            self.client = chromadb.Client()

        try:
            self.collection = self.client.get_or_create_collection("indian_law_sections")
        except Exception:
            self.collection = self.client.create_collection("indian_law_sections")

        # Count / index
        count = 0
        try:
            count = self.collection.count()
        except Exception:
            try:
                info = self.collection.get()
                count = len(info.get("ids", []))
            except Exception:
                count = 0

        if force_reindex or count == 0:
            print("Indexing laws into ChromaDB...")
            self.index_laws()
        else:
            print(f"Chroma collection has {count} items.")

        # Gemini client
        if self.gemini_api_key and HAS_GEMINI:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.gemini_client = None

    def index_laws(self):
        ids, embeddings, metadatas, documents = [], [], [], []
        for law in self.laws:
            lid = law.get("section")
            doc = (law.get("title", "") + " " + law.get("text", "")).strip()
            emb = self.model.encode(doc).tolist()
            ids.append(lid)
            embeddings.append(emb)
            metadatas.append(law)
            documents.append(doc)

        batch = 64
        for i in range(0, len(ids), batch):
            self.collection.add(
                ids=ids[i:i+batch],
                embeddings=embeddings[i:i+batch],
                metadatas=metadatas[i:i+batch],
                documents=documents[i:i+batch]
            )
        print(f"Indexed {len(ids)} laws into ChromaDB.")

    def find_relevant_laws(self, query_text: str, top_k: int = 5) -> List[Dict]:
        query_emb = self.model.encode(query_text).tolist()
        preferred_include = ["metadatas", "distances", "ids"]
        try:
            res = self.collection.query(query_embeddings=[query_emb], n_results=top_k, include=preferred_include)
        except ValueError:
            res = self.collection.query(query_embeddings=[query_emb], n_results=top_k)

        metadatas = []
        distances = []
        ids = []
        if isinstance(res, dict):
            if "metadatas" in res:
                metadatas = res["metadatas"][0] if isinstance(res["metadatas"], list) and res["metadatas"] else []
            if "distances" in res:
                distances = res["distances"][0] if isinstance(res["distances"], list) and res["distances"] else []
            if "ids" in res:
                ids = res["ids"][0] if isinstance(res["ids"], list) and res["ids"] else []
        else:
            raise RuntimeError("Unexpected response shape from ChromaDB query")

        results = []
        for i, md in enumerate(metadatas):
            entry = dict(md) if isinstance(md, dict) else {"text": str(md)}
            if i < len(ids):
                entry["_id"] = ids[i]
            if i < len(distances):
                entry["_distance"] = float(distances[i])
            results.append(entry)

        # If distances missing, compute local cosine sims for returned candidates
        if not distances and results:
            try:
                cand_texts = [r.get("text", "") + " " + r.get("title", "") for r in results]
                cand_embs = np.array(self.model.encode(cand_texts))
                q_emb = np.array(self.model.encode(query_text)).reshape(1, -1)
                sims = (cand_embs @ q_emb.T).squeeze() / (np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(q_emb) + 1e-12)
                for i, s in enumerate(sims):
                    results[i]["_sim"] = float(s)
            except Exception:
                pass

        return results

    def crossencoder_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Rerank using local CrossEncoder when available.
        Input: candidates list (metadata dicts). Output: same list sorted by score desc with '_score' set.
        """
        if not self.cross_encoder or not candidates:
            return candidates

        pair_inputs = []
        for c in candidates:
            text = (c.get("title","") + ". " + c.get("text","")).strip()
            pair_inputs.append((query, text))

        # Cross-encoder returns scores; higher = more relevant
        scores = self.cross_encoder.predict(pair_inputs)
        for c, s in zip(candidates, scores):
            c["_score"] = float(s)
        candidates.sort(key=lambda x: -x.get("_score", 0.0))
        return candidates

    def _gemini_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        # keep previous logic for gemini fallback
        if not getattr(self, "gemini_client", None):
            return self.crossencoder_rerank(query, candidates) if self.cross_encoder else self._simple_rerank(query, candidates)

        parts = [f"Scenario: {query}\n"]
        for i, c in enumerate(candidates, start=1):
            parts.append(f"Candidate {i}:\nSection: {c.get('section')}\nTitle: {c.get('title')}\nText: {c.get('text')}\n")
        parts.append("\nRank the candidates by relevance (most relevant first). Output only a JSON list like [{\"section\":\"IPC 123\",\"score\":0.9}, ...].")
        prompt = "\n".join(parts)

        try:
            resp = self.gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            raw = resp.candidates[0].content.parts[0].text
            import re, json
            m = re.search(r'(\[.*\])', raw, re.S)
            if not m:
                return self.crossencoder_rerank(query, candidates) if self.cross_encoder else self._simple_rerank(query, candidates)
            ranking = json.loads(m.group(1))
            score_map = {item["section"]: float(item.get("score", 0.0)) for item in ranking}
            for c in candidates:
                c["_score"] = score_map.get(c.get("section"), 0.0)
            candidates.sort(key=lambda x: -x.get("_score", 0.0))
            return candidates
        except Exception as e:
            print("[gemini rerank error]", e)
            return self.crossencoder_rerank(query, candidates) if self.cross_encoder else self._simple_rerank(query, candidates)

    def _simple_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return candidates
        cand_texts = [c.get("text", "") + " " + c.get("title", "") for c in candidates]
        cand_embs = np.array(self.model.encode(cand_texts))
        q_emb = np.array(self.model.encode(query)).reshape(1, -1)
        sims = (cand_embs @ q_emb.T).squeeze() / (np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(q_emb) + 1e-12)
        order = np.argsort(-sims)
        reranked = [candidates[i] for i in order]
        for idx, i in enumerate(order):
            reranked[idx]["_sim"] = float(sims[i])
        return reranked

    def retrieve_and_rerank(self, query: str, top_k: int = 6, reranker: str = "auto") -> List[Dict]:
        """
        High-level helper:
        - expand query (synonym expansion)
        - retrieve top_k dense results
        - rerank using: gemini (if requested & available) -> local cross-encoder -> simple sim
        reranker: "auto" (use gemini if present else crossencoder else simple), "cross" (force cross encoder),
                  "gemini" (force gemini), "simple" (force simple)
        """
        expanded = expand_query_simple(query, SYNONYMS)
        results = self.find_relevant_laws(expanded, top_k=top_k)

        if reranker == "gemini" and getattr(self, "gemini_client", None):
            return self._gemini_rerank(query, results)
        if reranker in ("auto", "cross") and self.cross_encoder:
            return self.crossencoder_rerank(query, results)
        return self._simple_rerank(query, results)

    def make_legal_explanation(self, query: str, matches: List[Dict], use_gemini: bool = True) -> str:
        ctx = []
        for m in matches:
            sec = m.get("section")
            title = m.get("title")
            text = m.get("text", "")
            ctx.append(f"{sec} - {title}\n{text}\n")
        prompt = f"Scenario:\n{query}\n\nRelevant sections:\n" + "\n---\n".join(ctx) + "\n\nIdentify applicable sections and a one-line rationale for each. If none clearly apply, reply: No strong match."
        if use_gemini and self.gemini_client:
            try:
                resp = self.gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                return resp.candidates[0].content.parts[0].text
            except Exception as e:
                print("[gemini generate error]", e)
        # fallback
        lines = [f"Scenario: {query}", "", "Top candidate sections and simple rationale:"]
        for m in matches:
            sec = m.get("section")
            title = m.get("title")
            text = m.get("text", "")
            ql = query.lower()
            kw_count = sum(1 for tok in set(text.lower().split()) if len(tok) > 4 and tok in ql)
            rationale = f"{'Several' if kw_count>=3 else 'Some' if kw_count>0 else 'No clear'} keyword overlaps ({kw_count})."
            lines.append(f"{sec} - {title}\nRationale: {rationale}\n")
        return "\n".join(lines)

    def force_reindex(self):
        try:
            self.client.delete_collection("indian_law_sections")
        except Exception:
            try:
                self.collection.delete()
            except Exception:
                pass
        try:
            self.collection = self.client.get_or_create_collection("indian_law_sections")
        except Exception:
            self.collection = self.client.create_collection("indian_law_sections")
        self.index_laws()
