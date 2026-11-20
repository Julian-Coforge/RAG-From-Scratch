from sentence_transformers import CrossEncoder


def make_reranker(model_name):
    """Load a CrossEncoder once and return a scorer(pairs) -> list of floats."""
    model = CrossEncoder(model_name)

    def scorer(pairs):
        """Score each (query, chunk) pair; higher means more relevant."""
        return [float(score) for score in model.predict(pairs)]

    return scorer


def rerank(query, results, scorer, top_n=3):
    """Re-score retrieved chunks against the query and return the top_n, best first."""
    if not results:
        return []

    # The cross-encoder reads query and chunk together, so it judges relevance more precisely than vector distance
    scores = scorer([(query, r["chunk"]) for r in results])

    # Build new dicts so the caller's original ranking is left untouched
    reranked = [{**r, "rerank_score": float(score)} for r, score in zip(results, scores)]
    reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
    return reranked[:top_n]
