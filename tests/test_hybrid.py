import pytest

from rag.hybrid import tokenize, keyword_search, rrf_fuse, hybrid_search
from rag.store import add_chunks, search


# ---------- tokenize ----------

def test_tokenize_lowercases():
    assert tokenize("Meal CAP") == ["meal", "cap"]


def test_tokenize_strips_punctuation():
    assert tokenize("$70, per day.") == ["70", "per", "day"]


def test_tokenize_keeps_hyphenated_ids_whole():
    assert tokenize("See FIN-204.") == ["see", "fin-204"]


# ---------- keyword_search (hand-built BM25) ----------

KW_CHUNKS = [
    {"id": "meals", "chunk": "The meal cap is $70 per day.", "source": "a.md"},
    {"id": "receipts", "chunk": "FIN-204 requires itemised receipts over $25.", "source": "a.md"},
    {"id": "travel", "chunk": "Book economy class for flights.", "source": "b.md"},
]


def test_keyword_search_ranks_exact_term_match_first():
    assert keyword_search(KW_CHUNKS, "FIN-204 receipts", k=3)[0]["id"] == "receipts"


def test_keyword_search_excludes_chunks_with_no_matching_terms():
    assert [r["id"] for r in keyword_search(KW_CHUNKS, "economy flights", k=3)] == ["travel"]


def test_keyword_search_with_no_matches_returns_empty_list():
    assert keyword_search(KW_CHUNKS, "zebra", k=3) == []


def test_keyword_search_returns_at_most_k_results():
    assert len(keyword_search(KW_CHUNKS, "cap receipts economy", k=2)) == 2


def test_keyword_search_result_includes_float_score():
    assert isinstance(keyword_search(KW_CHUNKS, "receipts", k=1)[0]["score"], float)


def test_keyword_search_result_keeps_chunk_fields():
    assert keyword_search(KW_CHUNKS, "receipts", k=1)[0]["source"] == "a.md"


IDF_CHUNKS = [
    {"id": "common", "chunk": "Policy policy policy overview."},
    {"id": "rare", "chunk": "FIN-204 overview."},
    {"id": "other", "chunk": "Policy summary."},
]


def test_keyword_search_rare_term_outranks_repeated_common_term():
    # "policy" is in 2 of 3 chunks (low IDF); "fin-204" is in 1 (high IDF).
    assert keyword_search(IDF_CHUNKS, "policy FIN-204", k=3)[0]["id"] == "rare"


# ---------- rrf_fuse (Reciprocal Rank Fusion) ----------

A, B, C = {"id": "a"}, {"id": "b"}, {"id": "c"}


def test_rrf_fuse_item_first_in_both_lists_ranks_first():
    assert rrf_fuse([[A, B], [A, B]])[0]["id"] == "a"


def test_rrf_fuse_item_in_both_lists_beats_item_top_of_one():
    assert rrf_fuse([[A, B], [C, B]])[0]["id"] == "b"


def test_rrf_fuse_does_not_duplicate_items():
    assert len(rrf_fuse([[A, B], [B, A]])) == 2


def test_rrf_fuse_score_is_one_over_k_plus_rank():
    assert rrf_fuse([[A]], k=60)[0]["rrf_score"] == pytest.approx(1 / 61)


def test_rrf_fuse_keeps_item_fields():
    assert rrf_fuse([[{"id": "a", "chunk": "text"}]])[0]["chunk"] == "text"


def test_rrf_fuse_of_no_lists_returns_empty_list():
    assert rrf_fuse([]) == []


# ---------- hybrid_search (vector + keyword, fused) ----------

# The fake embedder counts substrings, so "caps" looks like "cap" to it (fuzzy,
# like a real embedding). Keyword search matches whole tokens, so it only sees
# the exact ID "fin-204". Vector-only therefore picks the caps chunk; hybrid
# should lift the chunk with the exact ID match to the top.
HYBRID_CHUNKS = [
    {"id": "a.md::Meals::0", "chunk": "Daily caps apply.", "section": "Meals", "source": "a.md"},
    {"id": "a.md::Receipts::0", "chunk": "Receipts rules for FIN-204.", "section": "Receipts", "source": "a.md"},
    {"id": "a.md::Travel::0", "chunk": "Book economy class.", "section": "Travel", "source": "a.md"},
]
QUERY = "cap FIN-204"


def test_vector_only_misses_exact_id_match(collection, fake_embed):
    # Sanity check of the fixture: this is the failure hybrid fixes.
    add_chunks(collection, HYBRID_CHUNKS, fake_embed)
    assert search(collection, QUERY, fake_embed, k=1)[0]["id"] == "a.md::Meals::0"


def test_hybrid_search_ranks_exact_id_match_first(collection, fake_embed):
    add_chunks(collection, HYBRID_CHUNKS, fake_embed)
    assert hybrid_search(collection, QUERY, fake_embed, k=1)[0]["id"] == "a.md::Receipts::0"


def test_hybrid_search_returns_k_results(collection, fake_embed):
    add_chunks(collection, HYBRID_CHUNKS, fake_embed)
    assert len(hybrid_search(collection, QUERY, fake_embed, k=2)) == 2


def test_hybrid_search_result_keeps_metadata_for_attribution(collection, fake_embed):
    add_chunks(collection, HYBRID_CHUNKS, fake_embed)
    assert hybrid_search(collection, QUERY, fake_embed, k=1)[0]["section"] == "Receipts"


def test_hybrid_search_result_includes_chunk_text(collection, fake_embed):
    add_chunks(collection, HYBRID_CHUNKS, fake_embed)
    assert hybrid_search(collection, QUERY, fake_embed, k=1)[0]["chunk"] == "Receipts rules for FIN-204."
