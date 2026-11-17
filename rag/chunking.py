
def split_sections(markdown):
    """
    Splits a markdown string into sections based on headings. Each section is represented as a dictionary with 'section' and 'text' keys.
    """

    sections = []
    current = None

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
            if current is not None:
                sections.append(_finish_section(current))

            heading = stripped.lstrip("#").strip()

            # Handle headings that end with a '#' character, which may indicate a closing heading in some markdown styles. We remove the trailing '#' and any whitespace before it.
            if heading.endswith("#") and " #" in heading:
                heading = heading.rsplit(" #", 1)[0].rstrip()
            current = {"section": heading, "lines": []}

        # If the line is not a heading and there is a current section, add the line to the current section's lines.
        elif current is not None:
            current["lines"].append(line)

    # After processing all lines, if there is a current section, finalize it and add it to the sections list.
    if current is not None:
        sections.append(_finish_section(current))

    return sections

def _finish_section(current):
    """
    Finalises the current section by joining its lines into a single text string and returning a dictionary with the section name and text.
    """
    return {"section": current["section"], "text": "\n".join(current["lines"]).strip()}

