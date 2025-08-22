import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

class LawRetriever:
    def __init__(self, data_path: str):
        # Load environment variables
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env file")

        # Load IPC laws dataset
        with open(data_path, "r", encoding="utf-8") as f:
            self.laws = json.load(f)

        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Setup ChromaDB (persistent storage)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("indian_law_sections")

        # Index laws into vector DB (only if empty)
        if self.collection.count() == 0:
            self.index_laws()

        # Setup Gemini client
        self.gemini_client = genai.Client(api_key=gemini_api_key)

    def index_laws(self):
        for law in self.laws:
            emb = self.model.encode(law["text"]).tolist()
            self.collection.add(
                embeddings=[emb],
                metadatas=[law],
                ids=[law["section"]]
            )
        print(f"Indexed {len(self.laws)} laws into ChromaDB.")

    def find_relevant_laws(self, query_text: str, top_k: int = 3):
        query_emb = self.model.encode(query_text).tolist()
        res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        return res["metadatas"][0]  # List of dicts

    def make_legal_explanation(self, query: str, matches: list[dict]) -> str:
        # Prepare context
        context = "\n".join([
            f"{law['section']} - {law['title']}: {law['text']}" for law in matches
        ])

        prompt = (
            f"Crime Scene:\n{query}\n\n"
            f"Relevant IPC Sections:\n{context}\n\n"
            "Based on the above, which Indian law sections apply to the crime? "
            "Give section numbers, titles, and a short rationale."
        )

        # ✅ FIXED: use contents instead of input
        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        # Extract the generated text safely
        return response.candidates[0].content.parts[0].text
