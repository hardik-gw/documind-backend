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

        # Tokenize query cleanly into a set of unique words
        query_words = set(query.lower().split())
        reranked_results = []

        for chunk in chunks:
            # Handle both dictionary formats and potential objects safely
            if isinstance(chunk, dict):
                text_content = chunk.get("text", "").lower()
                base_score = chunk.get("similarity_score", 0.0)
                chunk_copy = chunk.copy()
            else:
                # Fallback safeguard in case objects are passed instead of dicts
                text_content = getattr(chunk, "page_content", "").lower() or getattr(chunk, "text", "").lower()
                base_score = getattr(chunk, "similarity_score", 0.0)
                chunk_copy = chunk.__dict__.copy() if hasattr(chunk, "__dict__") else {}

            # Calculate word match frequency overlap
            match_count = sum(1 for word in query_words if word in text_content)
            
            # Combine the vector similarity score with token frequency match weights
            final_score = base_score + (match_count * 0.1)
            
            chunk_copy["rerank_score"] = float(final_score)
            reranked_results.append(chunk_copy)

        # Sort by finalized density weights descending
        reranked_results = sorted(reranked_results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked_results[:top_k]