
def parse_frontmatter(markdown):
    """
    Parses the frontmatter of a markdown string, returning a tuple of (metadata dictionary, body string).
    """

    # Check if the markdown starts with frontmatter indicated by '---'. If it does, split the markdown into frontmatter and body. Otherwise, return an empty metadata dictionary and the original markdown as the body.
    if markdown.startswith("---"):
        parts = markdown.split("---", 2)

        # If there are at least three parts after splitting, the second part is treated as frontmatter and the third part as the body. The frontmatter is parsed line by line to extract key-value pairs, which are stored in a metadata dictionary. Lines without a colon are ignored, and leading/trailing whitespace and surrounding quotes on values are stripped.
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()
            meta = {}

            # Parse each line of the frontmatter to extract key-value pairs. Lines without a colon are ignored. Leading and trailing whitespace is stripped from keys and values, and any surrounding quotes on values are removed.
            for line in frontmatter.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip().strip('"').strip("'")  # Remove surrounding quotes if present
            return meta, body
    return {}, markdown

def split_sections(markdown):
    """
    Splits a markdown string into sections based on headings. Each section is represented as a dictionary with 'section' and 'text' keys.
    """

    sections = []
    current = {"section": "Preamble", "lines": []}

    for line in markdown.splitlines():

        stripped = line.strip()
        heading_level = len(stripped) - len(stripped.lstrip("#"))

        # Determine if the line is a heading based on the number of leading '#' characters
        is_heading = heading_level > 0 and (
            len(stripped) == heading_level
            or stripped[heading_level].isspace()
        )

        # If the line is a heading, start a new section; otherwise, add the line to the current section's text.
        if is_heading:
            _append_if_content(sections, current)

            heading = stripped.lstrip("#").strip()

            # Handle headings that end with a '#' character, which may indicate a closing heading in some markdown styles. We remove the trailing '#' and any whitespace before it.
            if heading.endswith("#") and " #" in heading:
                heading = heading.rsplit(" #", 1)[0].rstrip()
            current = {"section": heading, "lines": []}

        # If the line is not a heading and there is a current section, add the line to the current section's lines.
        else:
            current["lines"].append(line)

    # After processing all lines, if there is a current section, finalize it and add it to the sections list.
    _append_if_content(sections, current)


    return sections

def _finish_section(current):
    """
    Finalises the current section by joining its lines into a single text string and returning a dictionary with the section name and text.
    """
    return {"section": current["section"], "text": "\n".join(current["lines"]).strip()}

def _append_if_content(sections, current):
    """
    Appends the current section to the sections list if it contains any non-whitespace text.
    """
    if "\n".join(current["lines"]).strip():
        sections.append(_finish_section(current))

def chunk_text(text, size, overlap):
    """
    Splits a text string into chunks of a specified size with a specified overlap between chunks.
    """

    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()
    chunks = []
    start = 0

    # Loop through the words in the text, creating chunks of the specified size and overlap. The loop continues until all words have been processed. Each chunk is created by joining a slice of the words list from the current start index to the end index, which is determined by the specified size. If the end index reaches the end of the words list, the loop breaks to avoid creating an empty chunk. The start index is then updated to create the next chunk, taking into account the specified overlap.
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        # If the end index reaches the end of the words list, break the loop to avoid creating an empty chunk.
        if end == len(words):
            break        
        start += size - overlap

    return chunks

def chunk_document(markdown, source, size=180, overlap=30):
    """
    Splits a markdown document into chunks, attaching metadata such as section name, source, and policy ID to each chunk.

    Pipeline: parse_frontmatter -> split_sections -> chunk_text (word window).

    Chunking rationale:
    - Section-aware first: policy questions map onto policy sections ("how much
      PTO carries over?" -> Carryover), and the section name doubles as the
      source-attribution label.
    - size=180 words: all-MiniLM-L6-v2 truncates input at 256 tokens
      (~180-200 English words), so anything longer would be silently cut off
      before embedding.
    - overlap=30 words (~15%): keeps a sentence that straddles a window
      boundary intact in at least one chunk.
    - Words, not characters: no word is ever split in half.
    On the current corpus every section fits in a single chunk (largest is
    ~165 words), so the word window acts as a safety net for longer sections.

    Every frontmatter field (policy_id, version, status, superseded_by, ...) is
    copied onto each chunk so retrieved chunks can be traced to the exact
    document version they came from.
    """

    meta, body = parse_frontmatter(markdown)
    sections = split_sections(body)
    chunks = []

    # Iterate through each section of the document, chunking the text of each section and attaching metadata. Each chunk is a dictionary containing a unique id, the chunk text, section name, source, and every frontmatter field. Frontmatter is unpacked first so the pipeline's own keys win on a name clash.
    for section in sections:
        section_chunks = chunk_text(section["text"], size=size, overlap=overlap)
        for i, chunk in enumerate(section_chunks):
            chunk_dict = {
                **meta,
                "id": f"{source}::{section['section']}::{i}",
                "chunk": chunk,
                "section": section["section"],
                "source": source,
            }
            chunks.append(chunk_dict)

    return chunks