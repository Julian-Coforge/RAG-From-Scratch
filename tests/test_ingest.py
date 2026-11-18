from pathlib import Path

import pytest

from rag.ingest import build_index
from rag.store import make_embedder, search

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def _write_two_docs(folder):
    (folder / "a.md").write_text("# A\n## One\nAlpha text.")
    (folder / "b.md").write_text("# B\n## Two\nBeta text.")


def test_build_index_stores_chunks_from_every_doc(tmp_path, collection, fake_embed):
    _write_two_docs(tmp_path)
    build_index(tmp_path, collection, fake_embed)
    assert {m["source"] for m in collection.get()["metadatas"]} == {"a.md", "b.md"}


def test_build_index_returns_number_of_chunks_added(tmp_path, collection, fake_embed):
    _write_two_docs(tmp_path)
    assert build_index(tmp_path, collection, fake_embed) == 2


def test_build_index_ignores_non_markdown_files(tmp_path, collection, fake_embed):
    _write_two_docs(tmp_path)
    (tmp_path / "notes.txt").write_text("# Notes\n## Scratch\nIgnore me.")
    build_index(tmp_path, collection, fake_embed)
    assert collection.count() == 2


def test_build_index_on_empty_folder_returns_zero(tmp_path, collection, fake_embed):
    assert build_index(tmp_path, collection, fake_embed) == 0


def test_build_index_carries_frontmatter_metadata(tmp_path, collection, fake_embed):
    (tmp_path / "old.md").write_text("---\nstatus: superseded\n---\n# Old\n## Meals\nCap is $50.")
    build_index(tmp_path, collection, fake_embed)
    assert collection.get()["metadatas"][0]["status"] == "superseded"


def test_build_index_twice_does_not_duplicate_chunks(tmp_path, collection, fake_embed):
    _write_two_docs(tmp_path)
    build_index(tmp_path, collection, fake_embed)
    build_index(tmp_path, collection, fake_embed)
    assert collection.count() == 2


def test_build_index_accepts_folder_as_string(tmp_path, collection, fake_embed):
    _write_two_docs(tmp_path)
    assert build_index(str(tmp_path), collection, fake_embed) == 2


# ---------- real corpus + real model (slow) ----------

@pytest.mark.slow
def test_real_corpus_remote_work_question_retrieves_remote_work_policy(collection):
    embed = make_embedder("all-MiniLM-L6-v2")
    build_index(DOCS_DIR, collection, embed)
    top = search(collection, "How many days a week can I work from home?", embed, k=1)[0]
    assert top["source"] == "remote-work-policy.md"
