import os
import math
import json
import google.generativeai as genai

class VectorStoreService:
    def __init__(self):
        # Fetch key from environment safely
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY missing from environment variables.")
            
        clean_key = api_key.strip().strip('"').strip("'")
        genai.configure(api_key=clean_key)
        
        # Exact model string required by the legacy library structure
        self.model_name = "models/text-embedding-004"
        self.json_db_path = "./telegram_downloads/vault.json"
        self.vault = self._load_vault_from_disk()

    def _load_vault_from_disk(self) -> list:
        if os.path.exists(self.json_db_path):
            try:
                with open(self.json_db_path, "r") as f:
                    print("💾 Loading existing vault from disk...")
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error reading vault: {e}")
        return []

    def _save_vault_to_disk(self):
        try:
            os.makedirs(os.path.dirname(self.json_db_path), exist_ok=True)
            with open(self.json_db_path, "w") as f:
                json.dump(self.vault, f)
            print(f"💾 Vault synced to disk.")
        except Exception as e:
            print(f"❌ Failed to save vault: {e}")

    def get_embedding(self, text: str) -> list:
        try:
            # FIX: Execute proper legacy endpoint structural routing
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            # FIX: Correctly extract values array from object attribute instead of dict subscripting
            return response['embedding']
        except Exception as e:
            print(f"Embedding API Error: {e}")
            return [0.0] * 768

    def add_chunks(self, chunks: list, user_id: str):
        print(f"📦 Processing {len(chunks)} chunks...")
        self.vault = []
        
        for chunk in chunks:
            text_content = chunk.get("text", "")
            vector = self.get_embedding(text_content)
            self.vault.append({
                "user_id": str(user_id),
                "text": text_content,
                "vector": vector,
                "metadata": chunk.get("metadata", {})
            })
            
        self._save_vault_to_disk()
        print(f"✅ Committed {len(self.vault)} chunks to disk.")

    def query_similar_chunks(self, query: str, user_id: str, top_k: int = 5) -> list:
        self.vault = self._load_vault_from_disk()
        
        query_vector = self.get_embedding(query)
        user_docs = self.vault
        
        if not user_docs or not query_vector:
            print(f"⚠️ Vault empty. Count: {len(user_docs)}")
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