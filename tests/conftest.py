"""Shared pytest fixtures. pytest discovers this file automatically; tests
request a fixture by naming it as a parameter (no import needed)."""
import uuid

import chromadb
import pytest

# Fake embedder: one dimension per vocab word (its count in the text), plus a
# constant 1.0 so no vector is all zeros. Distances are then predictable:
# a query mentioning "cap" lands nearest the chunk that mentions "cap".
VOCAB = ["cap", "economy"]


def _fake_embed(texts):
    return [[float(t.lower().count(w)) for w in VOCAB] + [1.0] for t in texts]


@pytest.fixture
def fake_embed():
    return _fake_embed


@pytest.fixture
def collection():
    # EphemeralClient instances share one in-memory store per process,
    # so each test needs its own collection name to start empty.
    return chromadb.EphemeralClient().create_collection(f"test-{uuid.uuid4().hex}")
