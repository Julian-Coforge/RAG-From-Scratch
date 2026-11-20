from pathlib import Path

from rag.chunking import chunk_document
from rag.store import add_chunks


def build_index(folder, collection, embedder):
    """
    Builds an index of chunks from markdown files in the specified folder and adds them to a ChromaDB collection.
    """
    # Accept either a string or a Path for the folder
    folder = Path(folder)

    # Gather all chunks from the markdown files in the folder
    all_chunks = []
    for md_file in sorted(folder.glob("*.md")):
        all_chunks.extend(chunk_document(md_file.read_text(), source=md_file.name))

    # Add the chunks to the ChromaDB collection
    add_chunks(collection, all_chunks, embedder)

    return len(all_chunks)
