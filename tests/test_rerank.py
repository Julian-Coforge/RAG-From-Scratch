from pathlib import Path

import pytest

from rag.hybrid import hybrid_search
from rag.ingest import build_index
from rag.rerank import make_reranker, rerank
from rag.store import make_embedder

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RESULTS = [
    {"id": "a", "chunk": "Purpose and scope of expenses.", "source": "exp.md"},
    {"id": "b", "chunk": "IT supplies standard peripherals.", "source": "remote.md"},
    {"id": "c", "chunk": "Peripherals over $250 need peripherals approval.", "source": "exp.md"},
]


def _fake_scorer(pairs):
    # Score = how many times the chunk mentions "peripherals"
    return [float(chunk.lower().count("peripherals")) for _, chunk in pairs]


def _exploding_scorer(pairs):
    raise AssertionError("scorer should not be called with no results")


# ---------- rerank ----------

def test_rerank_puts_highest_scored_chunk_first():
    assert rerank("extra peripherals?", RESULTS, _fake_scorer)[0]["id"] == "c"


def test_rerank_orders_all_results_by_score():
    assert [r["id"] for r in rerank("q", RESULTS, _fake_scorer)] == ["c", "b", "a"]


def test_rerank_returns_at_most_top_n():
    assert len(rerank("q", RESULTS, _fake_scorer, top_n=1)) == 1


def test_rerank_with_top_n_larger_than_results_returns_all():
    assert len(rerank("q", RESULTS, _fake_scorer, top_n=10)) == 3


def test_rerank_result_includes_float_rerank_score():
    assert isinstance(rerank("q", RESULTS, _fake_scorer)[0]["rerank_score"], float)


def test_rerank_result_keeps_original_fields():
    assert rerank("q", RESULTS, _fake_scorer)[0]["source"] == "exp.md"


def test_rerank_passes_query_chunk_pairs_to_scorer():
    seen = []

    def spy(pairs):
        seen.extend(pairs)
        return [0.0] * len(pairs)

    rerank("my question", RESULTS, spy)
    assert seen[0] == ("my question", "Purpose and scope of expenses.")


def test_rerank_does_not_mutate_input_results():
    rerank("q", RESULTS, _fake_scorer)
    assert "rerank_score" not in RESULTS[0]


def test_rerank_with_no_results_does_not_call_scorer():
    rerank("q", [], _exploding_scorer)  # raises if the scorer is called


def test_rerank_with_no_results_returns_empty_list():
    assert rerank("q", [], _exploding_scorer) == []


# ---------- real cross-encoder (slow) ----------

@pytest.mark.slow
def test_make_reranker_returns_one_float_per_pair():
    scorer = make_reranker(RERANK_MODEL)
    assert all(isinstance(s, float) for s in scorer([("q", "a"), ("q", "b")]))


@pytest.mark.slow
def test_real_cross_encoder_scores_relevant_chunk_higher():
    scorer = make_reranker(RERANK_MODEL)
    relevant, irrelevant = scorer([
        ("How many days can I work from home?", "Employees may work remotely up to two days per week."),
        ("How many days can I work from home?", "Meal costs are reimbursed up to $70 per day."),
    ])
    assert relevant > irrelevant


@pytest.mark.slow
def test_real_corpus_rerank_puts_equipment_chunk_first(collection):
    # Hybrid alone ranks Equipment #2 for this query; reranking should lift it to #1
    query = "Can I get reimbursed for extra peripherals?"
    embed = make_embedder("all-MiniLM-L6-v2")
    build_index(DOCS_DIR, collection, embed)
    candidates = hybrid_search(collection, query, embed, k=10)
    top = rerank(query, candidates, make_reranker(RERANK_MODEL), top_n=3)[0]
    assert top["section"] == "Equipment and information security"
