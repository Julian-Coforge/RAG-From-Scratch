from pathlib import Path

from rag.chunking import chunk_document
from rag.store import add_chunks


def build_index(folder, collection, embedder):
    """
    Builds an index of chunks from markdown files in a specified folder and adds them to a ChromaDB collection.
    """

    

    # TODO: all_chunks = []
    # TODO: for md_file in sorted(Path(folder).glob("*.md")):
    #           all_chunks.extend(chunk_document(md_file.read_text(), source=md_file.name))
    # TODO: add_chunks(...) ONCE, then return len(all_chunks)

