"""
Command-line entry point: index the policy docs and retrieve the chunks most
relevant to a question.

Usage (from the project root):
    python -m rag.ask "How many days a week can I work from home?"
    python -m rag.ask "What is the meal allowance?" --k 5
"""
import argparse
from pathlib import Path

import chromadb

from rag.ingest import build_index
from rag.store import make_embedder, search

# Paths are resolved from the project root so the script works from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DB_DIR = PROJECT_ROOT / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "policies"


def get_collection():
    """Open (or create) the on-disk chromadb collection in chroma_db/."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def main():
    parser = argparse.ArgumentParser(description="Ask a question over the policy documents.")
    parser.add_argument("question", help="natural-language question")
    parser.add_argument("--k", type=int, default=3, help="number of chunks to retrieve (default 3)")
    args = parser.parse_args()

    # Load the embedding model once and reuse it for both indexing and the query
    embed = make_embedder(MODEL_NAME)

    # (Re)index the docs on every run. add_chunks upserts, so this refreshes
    # changed chunks without duplicating them; the corpus is small enough
    # (~20 chunks) that this takes about a second.
    collection = get_collection()
    count = build_index(DOCS_DIR, collection, embed)
    print(f"Indexed {count} chunks from {DOCS_DIR.name}/ into {DB_DIR.name}/ (collection size: {collection.count()})")

    # Retrieve the top-k chunks, closest (smallest distance) first
    results = search(collection, args.question, embed, k=args.k)

    print(f"\nQuestion: {args.question}\n")
    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] distance={r['distance']:.4f}  {r['source']} > {r['section']}"
              f"  (v{r.get('version', '?')}, {r.get('status', 'n/a')})")
        print(f"    {r['chunk'][:200]}{'...' if len(r['chunk']) > 200 else ''}\n")


if __name__ == "__main__":
    main()
