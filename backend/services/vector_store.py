import os
import chromadb
from typing import List, Dict, Any, Optional
from backend.services.embeddings import EmbeddingService

class VectorStoreService:
    def __init__(self, storage_path: str = "./chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=storage_path)
        self.embedder = EmbeddingService()
        self.collection = self.chroma_client.get_or_create_collection(name="document_chunks")

    def add_chunks(self, chunks: List[Dict[str, Any]], user_id: str = "guest_user"):
        """
        Saves document chunks permanently into ChromaDB, explicitly tagging
        every single record with a mandatory user_id metadata attribute.
        """
        if not chunks:
            print("⚠️ No chunks provided to save.")
            return

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        print(f"🔒 Storing {len(chunks)} chunks in ChromaDB isolated under User: '{user_id}'...")

        for idx, item in enumerate(chunks):
            text_content = item["text"]
            metadata_dict = item["metadata"]

            # Generate a globally unique ID including the user_id scope
            chunk_id = f"{user_id}_chunk_{idx}_{metadata_dict['source'].replace(' ', '_')}"
            vector = self.embedder.get_embedding(text_content)

            ids.append(chunk_id)
            embeddings.append(vector)
            documents.append(text_content)
            
            # Inject user_id directly into the metadata dictionary for hard filtering
            metadatas.append({
                "user_id": user_id,
                "source": metadata_dict["source"],
                "page": int(metadata_dict["page"])
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"✅ Isolated storage partition committed for '{user_id}'.")

    def query_similar_chunks(self, user_query: str, user_id: str = "guest_user", top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes a vector search, applying a strict metadata filter constraint
        ensuring ONLY chunks matching the specified user_id are evaluated.
        """
        query_vector = self.embedder.get_embedding(user_query)

        # 🔒 CRITICAL: The 'where' dictionary enforces a metadata tenancy wall inside ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"user_id": user_id} 
        )

        formatted_results = []
        if results and results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        
        return formatted_results