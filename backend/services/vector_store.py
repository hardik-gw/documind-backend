import os
import math
import json
from google import genai

class VectorStoreService:
    def __init__(self):
        api_key_env = os.environ.get("GEMINI_API_KEY")
        if not api_key_env:
            raise ValueError("❌ Error: GEMINI_API_KEY missing from environment.")
            
        clean_key = api_key_env.strip().strip('"').strip("'")
        
        self.client = genai.Client(api_key=clean_key)
        self.model_name = "models/text-embedding-004"
        
        self.json_db_path = "./telegram_downloads/vault.json"
        self.vault = self._load_vault_from_disk()

    def _load_vault_from_disk(self) -> list:
        if os.path.exists(self.json_db_path):
            try:
                with open(self.json_db_path, "r") as f:
                    print("💾 Found existing vault database file on disk. Loading chunks...")
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error reading vault from disk: {e}")
        return []

    def _save_vault_to_disk(self):
        try:
            os.makedirs(os.path.dirname(self.json_db_path), exist_ok=True)
            with open(self.json_db_path, "w") as f:
                json.dump(self.vault, f)
            print(f"💾 Database synced to disk at: {self.json_db_path}")
        except Exception as e:
            print(f"❌ Failed to save database to disk: {e}")

    def get_embedding(self, text: str) -> list:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Embedding API Error: {e}")
            return [0.0] * 768

    def add_chunks(self, chunks: list, user_id: str):
        print(f"📦 Processing Ingestion Core for {len(chunks)} fragments...")
        self.vault = []
        
        for chunk in chunks:
            text_content = chunk.get("text", "")
            vector = self.get_embedding(text_content)
            self.vault.append({
                "user_id": "MASTER_USER_BYPASS",  # 🎯 FORCE GLOBAL OVERRIDE KEY
                "text": text_content,
                "vector": vector,
                "metadata": chunk.get("metadata", {})
            })
            
        self._save_vault_to_disk()
        print(f"🧱 Successfully committed {len(self.vault)} chunks directly to disk database.")

    def query_similar_chunks(self, query: str, user_id: str, top_k: int = 5) -> list:
        self.vault = self._load_vault_from_disk()
        query_vector = self.get_embedding(query)
        
        # 🎯 FORCE SEARCH TO READ GLOBAL BYPASS KEY
        user_docs = [doc for doc in self.vault if doc["user_id"] == "MASTER_USER_BYPASS"]  
        
        if not user_docs or not query_vector:
            print(f"⚠️ Query Warning: Vault is empty or key mismatch occurred.")
            return []

        query_magnitude = math.sqrt(sum(q * q for q in query_vector))
        if query_magnitude == 0:
            return []

        scored_docs = []
        for doc in user_docs:
            doc_vector = doc["vector"]
            dot_product = sum(q * d for q, d in zip(query_vector, doc_vector))
            doc_magnitude = math.sqrt(sum(d * d for d in doc_vector))
            
            if doc_magnitude == 0:
                continue
                
            cosine_similarity = dot_product / (query_magnitude * doc_magnitude)
            normalized_score = (cosine_similarity + 1) / 2
            
            doc_copy = doc.copy()
            doc_copy["similarity_score"] = normalized_score
            scored_docs.append(doc_copy)

        scored_docs = sorted(scored_docs, key=lambda x: x["similarity_score"], reverse=True)
        return scored_docs[:top_k]