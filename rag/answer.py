

def build_prompt(question: str, results: list[dict]) -> str:
    """
    Build a prompt for the LLM that includes the question and the retrieved chunks.
    """

    # Start with the question
    prompt = f"Question: {question}\n\n"

    # Add each chunk's text and metadata to the prompt
    for i, result in enumerate(results, start=1):
        prompt += (
            f"[{i}] Source: {result.get('source', 'unknown')}, "
            f"Section: {result.get('section', 'unknown')}, "
            f"Version: {result.get('version', '?')}, "
            f"Status: {result.get('status', 'n/a')}\n"
            f"{result['chunk']}\n\n"
        )

    return prompt.strip()

def answer(question, results, llm_fn):
    """Ask the LLM to answer the question using only the retrieved results."""
    prompt = build_prompt(question, results)
    text = llm_fn(prompt)
    return {"text": text, "prompt": prompt}
