from rag.chunking import split_sections

MD = "# PTO Policy\nIntro text.\n## Accrual\nYou earn 1.5 days a month.\n## Carryover\nMax 5 days."

def test_split_sections_finds_section_names():
    assert [s["section"] for s in split_sections(MD)] == ["PTO Policy", "Accrual", "Carryover"]

def test_split_sections_keeps_section_body_text():
    assert split_sections(MD)[1]["text"] == "You earn 1.5 days a month."