from rag.answer import build_prompt, answer

CHUNKS = [
    {"chunk": "Cap is $70 per day.", "source": "a.md", "section": "Meals"},
    {"chunk": "Book economy class.", "source": "a.md", "section": "Travel"},
]

def test_build_prompt_includes_question():
    assert "What is the meal cap?" in build_prompt("What is the meal cap?", CHUNKS)

def test_build_prompt_includes_chunk_text():
    assert "Book economy class." in build_prompt("What is the meal cap?", CHUNKS)

def test_build_prompt_labels_chunks_with_source_and_section():
    assert "[2] Source: a.md, Section: Travel" in build_prompt("q", CHUNKS)

def _fake_llm(prompt):
    return "The cap is $70 [1]."

def test_answer_returns_llm_text():
    assert answer("What is the meal cap?", CHUNKS, _fake_llm)["text"] == "The cap is $70 [1]."

def test_answer_lists_sources_in_rank_order():
    assert answer("q", CHUNKS, _fake_llm)["sources"] == ["a.md > Meals", "a.md > Travel"]

def _exploding_llm(prompt):
    raise AssertionError("LLM should not be called when nothing was retrieved")

def test_answer_with_no_results_does_not_call_llm():
    answer("q", [], _exploding_llm)  # raises if llm_fn is called

def test_answer_with_no_results_says_it_could_not_find():
    assert "couldn't find" in answer("q", [], _exploding_llm)["text"].lower()
