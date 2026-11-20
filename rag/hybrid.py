import math
import re

from rag.store import search

# Lowercase words; hyphenated IDs like "fin-204" stay one token, punctuation is dropped.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Standard BM25 parameters: K1 = term-frequency saturation, B = length normalisation.
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text):
    """Lowercase text and split it into word tokens, dropping punctuation."""
    return TOKEN_PATTERN.findall(text.lower())


def keyword_search(chunks, query, k=5):
    """Rank chunks against a query with BM25 and return the top-k matches."""
    if not chunks:
        return []

    # Preprocess the chunks and query
    docs = [tokenize(chunk["chunk"]) for chunk in chunks]
    n_docs = len(docs)
    avg_len = sum(len(doc) for doc in docs) / n_docs
    query_terms = set(tokenize(query))
    doc_freq = {term: sum(term in doc for doc in docs) for term in query_terms}

    # Score each chunk with BM25 and keep only those with a positive score
    scored = []
    for chunk, doc in zip(chunks, docs):
        score = 0.0
        for term in query_terms:
            tf = doc.count(term)
            if tf == 0:
                continue

            # Rare terms get a higher IDF, so "fin-204" outweighs "policy"
            idf = math.log((n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1)
            length_norm = 1 - BM25_B + BM25_B * len(doc) / avg_len
            score += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * length_norm)
        if score > 0:
            scored.append({**chunk, "score": score})

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:k]


def rrf_fuse(rankings, k=60):
    """Merge ranked lists with Reciprocal Rank Fusion: score = sum(1 / (k + rank))."""
    fused = {}

    # For each ranking, iterate through the items and update their RRF scores in the fused dictionary.
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            if item["id"] not in fused:
                fused[item["id"]] = {**item, "rrf_score": 0.0}
            fused[item["id"]]["rrf_score"] += 1 / (k + rank)

    # Sort the fused items by their RRF scores in descending order and return the sorted list.
    fused_list = list(fused.values())
    fused_list.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused_list


def _all_chunks(collection):
    """Read every stored chunk back out of chromadb as a chunk dict."""
    got = collection.get()
    return [
        {**metadata, "id": id_, "chunk": doc}
        for id_, doc, metadata in zip(got["ids"], got["documents"], got["metadatas"])
    ]


def hybrid_search(collection, query, embed_function, k=5, candidates=20):
    """Fuse vector and BM25 rankings and return the top-k chunks."""
    
    # Take more candidates than k so a chunk ranked low by one retriever can still win
    vector_results = search(collection, query, embed_function, k=candidates)
    keyword_results = keyword_search(_all_chunks(collection), query, k=candidates)

    # Use RRF's default constant, not k (k here is the result count)
    return rrf_fuse([vector_results, keyword_results])[:k]
