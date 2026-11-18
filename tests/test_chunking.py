from rag.chunking import split_sections, parse_frontmatter, chunk_text, chunk_document
import pytest

MD = "# PTO Policy\nIntro text.\n## Accrual\nYou earn 1.5 days a month.\n## Carryover\nMax 5 days."

def test_split_sections_finds_section_names():
    assert [s["section"] for s in split_sections(MD)] == ["PTO Policy", "Accrual", "Carryover"]

def test_split_sections_keeps_section_body_text():
    assert split_sections(MD)[1]["text"] == "You earn 1.5 days a month."

def test_split_sections_keeps_text_before_first_heading():
    md = "Effective 2026-01-01.\n# PTO Policy\nBody."
    assert split_sections(md)[0] == {"section": "Preamble", "text": "Effective 2026-01-01."}

def test_split_sections_skips_heading_with_no_body():
    md = "# Title\n## Accrual\nYou earn 1.5 days."
    assert [s["section"] for s in split_sections(md)] == ["Accrual"]

def test_parse_frontmatter_reads_policy_id():
    md = "---\npolicy_id: FIN-204\n---\n# Expenses\nBody."
    meta, body = parse_frontmatter(md)
    assert meta["policy_id"] == "FIN-204"

def test_parse_frontmatter_strips_quotes_from_values():
    md = '---\nversion: "1.4"\n---\nBody.'
    meta, _ = parse_frontmatter(md)
    assert meta["version"] == "1.4"

def test_parse_frontmatter_returns_body_after_fence():
    md = "---\npolicy_id: FIN-204\n---\n# Expenses\nBody."
    _, body = parse_frontmatter(md)
    assert body == "# Expenses\nBody."

def test_parse_frontmatter_without_frontmatter_returns_input_unchanged():
    md = "# Expenses\nBody."
    assert parse_frontmatter(md) == ({}, md)

def test_chunk_text_short_text_gives_one_chunk():
    assert chunk_text("one two three", size=5, overlap=1) == ["one two three"]

def test_chunk_text_no_extra_chunk_when_window_reaches_end():
    assert chunk_text("a b c d e", size=5, overlap=1) == ["a b c d e"]

def test_chunk_text_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=3, overlap=3)

def test_chunk_text_nothing_lost():
    words = " ".join(f"w{i}" for i in range(12))
    assert chunk_text(words, size=5, overlap=1)[-1].split()[-1] == "w11"

WORDS_12 = " ".join(f"w{i}" for i in range(12))

def test_chunk_text_no_chunk_exceeds_size():
    assert all(len(c.split()) <= 5 for c in chunk_text(WORDS_12, size=5, overlap=1))

def test_chunk_text_neighbouring_chunks_share_overlap():
    chunks = chunk_text(WORDS_12, size=5, overlap=1)
    assert chunks[0].split()[-1:] == chunks[1].split()[:1]

DOC = "---\npolicy_id: FIN-204\nversion: \"1.4\"\n---\n# Expenses\n## Meals\nCap is $50 per day."

def test_chunk_document_attaches_section_name():
    assert chunk_document(DOC, source="expenses-v1.4.md")[0]["section"] == "Meals"

def test_chunk_document_attaches_source():
    assert chunk_document(DOC, source="expenses-v1.4.md")[0]["source"] == "expenses-v1.4.md"

def test_chunk_document_attaches_policy_id():
    assert chunk_document(DOC, source="expenses-v1.4.md")[0]["policy_id"] == "FIN-204"

def test_chunk_document_attaches_version():
    assert chunk_document(DOC, source="expenses-v1.4.md")[0]["version"] == "1.4"

def test_chunk_document_ids_are_unique():
    long_doc = "# Policy\n## Meals\n" + " ".join(f"w{i}" for i in range(400))
    ids = [c["id"] for c in chunk_document(long_doc, source="x.md")]
    assert len(ids) == len(set(ids))
