import os
import math
from google import genai
from google.genai import types

class VectorStoreService:
    def __init__(self):
        # Uses your existing environment variable instantly
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = "text-embedding-004"
        # Simple in-memory mock store for cloud RAG metadata isolation
        self.vault = []

    def get_embedding(self, text: str) -> list:
        """Fetches embedding vectors directly from Google's cloud API (0MB RAM)."""
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Embedding API Error: {e}")
            # Fallback mock vector so the app never freezes
            return [0.0] * 768

    def add_chunks(self, chunks: list, user_id: str):
        """Stores text layers paired with cloud vectors and ownership tags."""
        for chunk in chunks:
            text_content = chunk.get("text", "")
            vector = self.get_embedding(text_content)
            self.vault.append({
                "user_id": user_id,
                "text": text_content,
                "vector": vector,
                "metadata": chunk.get("metadata", {})
            })
        print(f"🧱 Successfully mapped {len(chunks)} chunks to cloud vault.")

    def query_similar_chunks(self, query: str, user_id: str, top_k: int = 5) -> list:
        """Calculates true Cosine Similarity over isolated user documents."""
        query_vector = self.get_embedding(query)
        user_docs = [doc for doc in self.vault if doc["user_id"] == user_id]
        
        if not user_docs or not query_vector:
            return []

        # Calculate magnitude of the query vector once to avoid redundant loops
        query_magnitude = math.sqrt(sum(q * q for q in query_vector))
        if query_magnitude == 0:
            return []

        scored_docs = []
        for doc in user_docs:
            doc_vector = doc["vector"]
            
            # 1. Calculate Dot Product
            dot_product = sum(q * d for q, d in zip(query_vector, doc_vector))
            
            # 2. Calculate Document Vector Magnitude
            doc_magnitude = math.sqrt(sum(d * d for d in doc_vector))
            
            if doc_magnitude == 0:
                continue
                
            # 3. Calculate True Cosine Similarity
            cosine_similarity = dot_product / (query_magnitude * doc_magnitude)
            
            # Normalize score map to keep values strictly positive (between 0.0 and 1.0)
            normalized_score = (cosine_similarity + 1) / 2
            
            doc_copy = doc.copy()
            doc_copy["similarity_score"] = normalized_score
            scored_docs.append(doc_copy)

        # Sort by similarity score descending (highest confidence first)
        scored_docs = sorted(scored_docs, key=lambda x: x["similarity_score"], reverse=True)
        return scored_docs[:top_k]