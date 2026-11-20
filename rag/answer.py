import anthropic

MODEL = "claude-opus-5-5"

# Rules for every answer. The retrieved chunks arrive in the user message,
# each labelled "[n] Source: ..., Section: ..., Version: ..., Status: ...".
SYSTEM_PROMPT = """You answer employee questions about Northbridge Systems company policy.

Each question comes with numbered context chunks taken from the policy documents. Base your answer only on those chunks. Do not use outside knowledge, and do not guess. Employees act on these answers, so a confident wrong figure is worse than no answer.

Cite the chunk that supports each fact inline, using its number in square brackets, for example: "The daily meal cap is $70 [2]." Cite only chunks that actually contain the fact.

If the chunks do not contain the answer, say "I couldn't find that in the policy documents." and stop. Do not fill the gap with a plausible-sounding answer.

Each chunk shows a Status. If chunks disagree, for example two versions of the same policy give different numbers, base your answer on the chunk marked "active". Then say briefly that a superseded version gives a different figure, and name both versions, so the employee knows a conflict exists.

Keep answers short: give the direct answer first, then any condition or exception that matters. Use plain sentences, not headings."""

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
    if not results:
                return {"text": "Couldn't find any relevant information to answer your question.", "prompt": None, "sources": []}
    else:
        prompt = build_prompt(question, results)
        text = llm_fn(prompt)
        sources = [f"{r.get('source', 'unknown')} > {r.get('section', 'unknown')}" for r in results]
        return {"text": text, "prompt": prompt, "sources": sources}


def make_claude_llm(model=MODEL):
    """Return an llm_fn(prompt) -> str that calls Claude with SYSTEM_PROMPT."""
    client = anthropic.Anthropic()

    def llm_fn(prompt):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        # If the model returns no content, provide a default message
        if response.stop_reason == "refusal":
            return "The model declined to answer this question."

        # Extract the text blocks from the response and join them into a single string
        text_blocks = [block.text for block in response.content if block.type == "text"]

        return "\n".join(text_blocks).strip()

    return llm_fn
