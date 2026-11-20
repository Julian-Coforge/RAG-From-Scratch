def parse_frontmatter(markdown):
    """Split a markdown string into (metadata dict, body) using its '---' frontmatter."""
    if markdown.startswith("---"):
        parts = markdown.split("---", 2)

        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()
            meta = {}

            # Parse "key: value" lines; split on the first colon only and strip quotes
            for line in frontmatter.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip().strip('"').strip("'")
            return meta, body
    return {}, markdown


def split_sections(markdown):
    """Split markdown into [{"section", "text"}] dicts, one per heading."""
    sections = []
    current = {"section": "Preamble", "lines": []}

    for line in markdown.splitlines():
        stripped = line.strip()
        heading_level = len(stripped) - len(stripped.lstrip("#"))

        # A heading is leading '#'s followed by whitespace or end of line ("#tag" is not)
        is_heading = heading_level > 0 and (
            len(stripped) == heading_level
            or stripped[heading_level].isspace()
        )

        if is_heading:
            _append_if_content(sections, current)
            heading = stripped.lstrip("#").strip()

            # Drop optional closing hashes: "## Meals ##" -> "Meals"
            if heading.endswith("#") and " #" in heading:
                heading = heading.rsplit(" #", 1)[0].rstrip()
            current = {"section": heading, "lines": []}
        else:
            current["lines"].append(line)

    _append_if_content(sections, current)
    return sections


def _finish_section(current):
    """Join a section's lines into its final {"section", "text"} form."""
    return {"section": current["section"], "text": "\n".join(current["lines"]).strip()}


def _append_if_content(sections, current):
    """Append the section only if it has non-blank text."""
    if "\n".join(current["lines"]).strip():
        sections.append(_finish_section(current))


def chunk_text(text, size, overlap):
    """Split text into windows of `size` words, each sharing `overlap` words with the last."""
    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        # Stop once a window reaches the last word, so no redundant tail chunk is made
        if end == len(words):
            break
        start += size - overlap

    return chunks


def chunk_document(markdown, source, size=180, overlap=30):
    """
    Chunk a policy document: parse_frontmatter -> split_sections -> chunk_text.

    Rationale: split by section first (sections match policy questions and give the
    attribution label); size=180 words stays under all-MiniLM-L6-v2's 256-token limit;
    overlap=30 words (~15%) keeps boundary sentences whole; words, not characters, so
    no word is cut. Every frontmatter field is copied onto each chunk for lineage.
    """
    meta, body = parse_frontmatter(markdown)
    sections = split_sections(body)
    chunks = []

    for section in sections:
        section_chunks = chunk_text(section["text"], size=size, overlap=overlap)
        for i, chunk in enumerate(section_chunks):
            # Frontmatter first so the pipeline's own keys win on a name clash
            chunk_dict = {
                **meta,
                "id": f"{source}::{section['section']}::{i}",
                "chunk": chunk,
                "section": section["section"],
                "source": source,
            }
            chunks.append(chunk_dict)

    return chunks
