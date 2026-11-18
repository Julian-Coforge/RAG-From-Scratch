from pathlib import Path

from rag.chunking import chunk_document
from rag.store import add_chunks


def build_index(folder, collection, embedder):
    """
    Builds an index of chunks from markdown files in the specified folder and adds them to a ChromaDB collection.

    Args:
        folder: The path to the folder containing markdown files (str or Path).
        collection: The ChromaDB collection to which the chunks will be added.
        embedder: A function that takes a list of texts and returns their embeddings.
    Returns:
        The number of chunks added to the collection.
    """
    # Accept either a string or a Path for the folder
    folder = Path(folder)

    # Chunk every markdown file using the tested chunking pipeline
    # (frontmatter -> sections -> word windows), so each chunk carries its
    # source, section and frontmatter metadata. Files are sorted so ingest
    # order is deterministic across runs and machines; non-.md files are skipped.
    all_chunks = []
    for md_file in sorted(folder.glob("*.md")):
        all_chunks.extend(chunk_document(md_file.read_text(), source=md_file.name))

    # Embed and store everything in one call. add_chunks upserts (so rebuilding
    # does not duplicate chunks) and is a no-op for an empty list.
    add_chunks(collection, all_chunks, embedder)

    return len(all_chunks)
