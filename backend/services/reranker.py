from sentence_transformers import CrossEncoder

_model = None

def get_reranker_model():
    global _model
    if _model is None:
        # Load lazily only when needed
        _model = CrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2") 
    return _model

def rerank_documents(query: str, documents: list, top_n: int = 3):
    if not documents:
        return []
    
    model = get_reranker_model()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)
    
    # Pair documents with their scores
    for doc, score in zip(documents, scores):
        doc.metadata["rerank_score"] = float(score)
        
    # Sort by score descending
    sorted_docs = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)
    return sorted_docs[:top_n]