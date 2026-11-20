import os
import uuid
from pathlib import Path

import chromadb
import pytest

from rag.answer import answer, make_claude_llm
from rag.chunking import chunk_document
from rag.eval_set import EVAL_SET
from rag.hybrid import hybrid_search
from rag.ingest import build_index
from rag.rerank import make_reranker, rerank
from rag.store import make_embedder

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
TOP_K = 3
CANDIDATES = 10

ANSWERABLE = [item for item in EVAL_SET if item["expected_source"]]
EXPENSE = [item for item in ANSWERABLE if item["expected_source"].startswith("expense")]
FACT_CASES = [(item["question"], fact) for item in EVAL_SET for fact in item["key_facts"]]

needs_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


def _qid(item):
    return item["question"][:45]


# ---------- eval set sanity (fast) ----------

def test_eval_set_has_at_least_eight_questions():
    assert len(EVAL_SET) >= 8


def test_eval_set_includes_an_unanswerable_question():
    assert any(item["expected_source"] is None for item in EVAL_SET)


@pytest.mark.parametrize("item", ANSWERABLE, ids=_qid)
def test_expected_section_exists_in_corpus(item):
    # Guards against typos in the eval set silently making recall tests unpassable
    text = (DOCS_DIR / item["expected_source"]).read_text()
    assert item["expected_section"] in {c["section"] for c in chunk_document(text, item["expected_source"])}


# ---------- full pipeline fixtures (built once per module) ----------

@pytest.fixture(scope="module")
def retrieve():
    """Index the real docs once; return retrieve(question) = hybrid top 10 -> rerank top 3."""
    embed = make_embedder("all-MiniLM-L6-v2")
    scorer = make_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    collection = chromadb.EphemeralClient().create_collection(f"eval-{uuid.uuid4().hex}")
    build_index(DOCS_DIR, collection, embed)

    def _retrieve(question):
        candidates = hybrid_search(collection, question, embed, k=CANDIDATES)
        return rerank(question, candidates, scorer, top_n=TOP_K)

    return _retrieve


@pytest.fixture(scope="module")
def answer_text(retrieve):
    """Return answer_text(question); each question hits the API once, however many facts it checks."""
    llm = make_claude_llm()
    cache = {}

    def _answer_text(question):
        if question not in cache:
            cache[question] = answer(question, retrieve(question), llm)["text"]
        return cache[question]

    return _answer_text


# ---------- retrieval recall (slow, no API calls) ----------

@pytest.mark.slow
@pytest.mark.parametrize("item", ANSWERABLE, ids=_qid)
def test_retrieval_recall_at_3(retrieve, item):
    hits = [(r["source"], r["section"]) for r in retrieve(item["question"])]
    assert (item["expected_source"], item["expected_section"]) in hits


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="Planted data issue: superseded expense v1.4 is still indexed and retrieved")
@pytest.mark.parametrize("item", EXPENSE, ids=_qid)
def test_superseded_policy_is_not_retrieved(retrieve, item):
    assert all(r.get("status") != "superseded" for r in retrieve(item["question"]))


# ---------- answer key facts (slow + paid API calls) ----------

@pytest.mark.slow
@pytest.mark.eval
@needs_api_key
@pytest.mark.parametrize("question,fact", FACT_CASES, ids=[f"{q[:35]} -> {f}" for q, f in FACT_CASES])
def test_answer_contains_key_fact(answer_text, question, fact):
    # Normalise curly apostrophes so "couldn’t find" matches "couldn't find"
    assert fact.lower() in answer_text(question).lower().replace("’", "'")
