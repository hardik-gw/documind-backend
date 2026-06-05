import os
import math
import json
from google import genai

class VectorStoreService:
    def __init__(self):
        api_key_env = os.environ.get("GEMINI_API_KEY")
        if not api_key_env:
            raise ValueError("❌ Error: GEMINI_API_KEY missing from environment variables.")
            
        clean_key = api_key_env.strip().strip('"').strip("'")
        self.client = genai.Client(api_key=clean_key)
        self.model_name = "models/text-embedding-004"
        
        # Writable file location on Render's operating system container
        self.json_db_path = "/tmp/telegram_downloads/vault.json"

    def _load_vault_from_disk(self) -> list:
        """Forcefully reads your data directly from the system storage layer."""
        if os.path.exists(self.json_db_path):
            try:
                with open(self.json_db_path, "r") as f:
                    print("💾 Read operation success: Loading chunks from secure /tmp directory.")
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Disk read exception error: {e}")
        return []

    def _save_vault_to_disk(self, data: list):
        """Forcefully dumps your data directly into the system storage layer."""
        try:
            os.makedirs(os.path.dirname(self.json_db_path), exist_ok=True)
            with open(self.json_db_path, "w") as f:
                json.dump(data, f)
            print(f"💾 Write operation success: Synced vault database to path: {self.json_db_path}")
        except Exception as e:
            print(f"❌ Disk write exception failure: {e}")

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
        print(f"📦 Commencing document ingestion loop for {len(chunks)} text layers...")
        
        # Load any existing documents out of the file first to preserve state
        current_vault = self._load_vault_from_disk()
        
        for chunk in chunks:
            text_content = chunk.get("text", "")
            vector = self.get_embedding(text_content)
            current_vault.append({
                "user_id": "GLOBAL_BYPASS_KEY",  # Forces cross-session synchronization
                "text": text_content,
                "vector": vector,
                "metadata": chunk.get("metadata", {})
            })
            
        self._save_vault_to_disk(current_vault)
        print(f"🧱 Successfully locked {len(current_vault)} total chunks onto the server disk.")

    def query_similar_chunks(self, query: str, user_id: str, top_k: int = 5) -> list:
        # 🎯 THE FIX: Reload chunks directly from disk on every single query call
        fresh_vault = self._load_vault_from_disk()
        query_vector = self.get_embedding(query)
        
        # Pull chunks using the synchronized bypass key layout
        user_docs = [doc for doc in fresh_vault if doc["user_id"] == "GLOBAL_BYPASS_KEY"]  
        
        print(f"🔍 Database Scanning: Found {len(user_docs)} chunks matched inside disk storage.")
        
        if not user_docs or not query_vector:
            print(f"⚠️ Query alert notice: Search targeted an empty file array block.")
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