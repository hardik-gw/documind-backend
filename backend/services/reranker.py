from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class RerankerService:
    def __init__(self):
        """
        Initializes a lightweight, high-performance Cross-Encoder model.
        The model downloads once on startup and runs entirely locally.
        """
        print("🧠 Loading local Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)...")
        # This specific model is fine-tuned explicitly for web search and document QA relevance
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Reranker model loaded and active.")

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Takes a query and a broad list of retrieved chunks, evaluates their true contextual 
        relevance scores simultaneously, and returns the top K reordered records.
        """
        if not chunks:
            return []

        # 1. Format pairs exactly how the Cross-Encoder expects: [[query, text1], [query, text2], ...]
        pairs = [[query, chunk["text"]] for chunk in chunks]

        # 2. Compute the relevance scores (higher score = more relevant)
        scores = self.model.predict(pairs)

        # 3. Attach the calculation scores to our chunk dictionaries
        for idx, score in enumerate(scores):
            chunks[idx]["rerank_score"] = float(score)

        # 4. Sort the list of chunks in descending order based on their new score
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        print(f"📊 Reranker processed {len(chunks)} fragments. Top score: {chunks[0]['rerank_score']:.4f}")
        
        # 5. Slice and return only the best top_k matching elements
        return chunks[:top_k]