class RerankerService:
    def __init__(self):
        pass

    def rerank(self, query: str, chunks: list, top_k: int = 3) -> list:
        """
        Lightweight lexical reranking engine (0MB RAM).
        Cross-scores semantic results against specific keyword density matrices.
        """
        if not chunks:
            return []

        query_words = set(query.lower().split())
        reranked_results = []

        for chunk in chunks:
            text_content = chunk.get("text", "").lower()
            # Calculate word match frequency overlap
            match_count = sum(1 for word in query_words if word in text_content)
            
            # Combine the vector similarity score with token frequency match weights
            base_score = chunk.get("similarity_score", 0.0)
            final_score = base_score + (match_count * 0.1)
            
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = float(final_score)
            reranked_results.append(chunk_copy)

        # Sort by finalized density weights descending
        reranked_results = sorted(reranked_results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_results[:top_k]