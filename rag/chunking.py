
def parse_frontmatter(markdown):
    """
    Parses the frontmatter of a markdown string, returning a tuple of (metadata dictionary, body string).
    """

    # Check if the markdown starts with frontmatter indicated by '---'. If it does, split the markdown into frontmatter and body. Otherwise, return an empty metadata dictionary and the original markdown as the body.
    if markdown.startswith("---"):
        parts = markdown.split("---", 2)
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