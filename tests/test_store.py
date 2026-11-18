import uuid
import chromadb
import pytest

from rag.store import add_chunks, search, make_embedder

# Fake embedder: one dimension per vocab word (its count in the text), plus a
# constant 1.0 so no vector is all zeros. Distances are then predictable:
# a query mentioning "cap" lands nearest the chunk that mentions "cap".
VOCAB = ["cap", "economy"]


def fake_embed(texts):
    return [[float(t.lower().count(w)) for w in VOCAB] + [1.0] for t in texts]


CHUNKS = [
    {"id": "a.md::Meals::0", "chunk": "Cap is $50 per day.", "section": "Meals",
     "source": "a.md", "policy_id": "FIN-204", "status": "superseded"},
    {"id": "a.md::Travel::0", "chunk": "Book economy class.", "section": "Travel",
     "source": "a.md", "policy_id": "FIN-204", "status": "superseded"},
]


@pytest.fixture
def collection():
    # EphemeralClient instances share one in-memory store per process,
    # so each test needs its own collection name to start empty.
    return chromadb.EphemeralClient().create_collection(f"test-{uuid.uuid4().hex}")


def _stored_metadata(collection, chunk_id):
    return collection.get(ids=[chunk_id])["metadatas"][0]


# ---------- add_chunks ----------

def test_add_chunks_stores_every_chunk(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert collection.count() == 2


def test_add_chunks_stores_chunk_text_as_document(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert collection.get(ids=["a.md::Meals::0"])["documents"][0] == "Cap is $50 per day."


def test_add_chunks_keeps_section_metadata(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert _stored_metadata(collection, "a.md::Meals::0")["section"] == "Meals"


def test_add_chunks_keeps_frontmatter_metadata(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert _stored_metadata(collection, "a.md::Meals::0")["status"] == "superseded"


def test_add_chunks_does_not_duplicate_text_into_metadata(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert "chunk" not in _stored_metadata(collection, "a.md::Meals::0")


def test_add_chunks_with_no_chunks_stores_nothing(collection):
    add_chunks(collection, [], fake_embed)
    assert collection.count() == 0


def test_add_chunks_twice_updates_instead_of_ignoring(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    edited = [{**CHUNKS[0], "chunk": "Cap is $75 per day."}]
    add_chunks(collection, edited, fake_embed)
    assert collection.get(ids=["a.md::Meals::0"])["documents"][0] == "Cap is $75 per day."


# ---------- search ----------

def test_search_returns_k_results(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert len(search(collection, "what is the cap", fake_embed, k=1)) == 1


def test_search_ranks_closest_chunk_first(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert search(collection, "what is the cap", fake_embed, k=2)[0]["id"] == "a.md::Meals::0"


def test_search_result_includes_chunk_text(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert search(collection, "what is the cap", fake_embed, k=1)[0]["chunk"] == "Cap is $50 per day."


def test_search_result_includes_metadata_for_attribution(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert search(collection, "what is the cap", fake_embed, k=1)[0]["section"] == "Meals"


def test_search_result_includes_distance(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert isinstance(search(collection, "what is the cap", fake_embed, k=1)[0]["distance"], float)


def test_search_with_k_larger_than_collection_returns_all(collection):
    add_chunks(collection, CHUNKS, fake_embed)
    assert len(search(collection, "what is the cap", fake_embed, k=10)) == 2


# ---------- real model (slow: loads all-MiniLM-L6-v2) ----------

@pytest.mark.slow
def test_real_embedder_retrieves_semantically_related_chunk(collection):
    embed = make_embedder("all-MiniLM-L6-v2")
    add_chunks(collection, CHUNKS, embed)
    assert search(collection, "How much can I spend on food each day?", embed, k=1)[0]["section"] == "Meals"
