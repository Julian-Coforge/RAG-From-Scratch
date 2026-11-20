import argparse
from pathlib import Path

import chromadb

from rag.ingest import build_index
from rag.store import make_embedder, search
from rag.answer import answer, make_claude_llm
from rag.hybrid import hybrid_search


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
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid",
                    help="retrieval method (default hybrid)")
    args = parser.parse_args()

    # Load the embedding model once and reuse it for both indexing and the query
    embed = make_embedder(MODEL_NAME)

    # Build the index (if not already built) and open the collection
    collection = get_collection()
    count = build_index(DOCS_DIR, collection, embed)
    print(f"Indexed {count} chunks from {DOCS_DIR.name}/ into {DB_DIR.name}/ (collection size: {collection.count()})")

    # Search for the most relevant chunks based on the question and the specified mode
    if args.mode == "hybrid":
        results = hybrid_search(collection, args.question, embed, k=args.k)
    else:
        results = search(collection, args.question, embed, k=args.k)

    print(f"\nMode: {args.mode}\nQuestion: {args.question}\n")

    # Print the retrieved chunks with their rank, score, source, section, version, status, and a preview of the chunk text.
    # Hybrid ranks by fused RRF score; keyword-only hits have no distance.
    for rank, r in enumerate(results, start=1):
        score = f"rrf={r['rrf_score']:.4f}" if args.mode == "hybrid" else f"distance={r['distance']:.4f}"
        print(f"[{rank}] {score}  {r['source']} > {r['section']}"
              f"  (v{r.get('version', '?')}, {r.get('status', 'n/a')})")
        print(f"    {r['chunk'][:200]}{'...' if len(r['chunk']) > 200 else ''}\n")

    # Ask the LLM to answer the question using the retrieved chunks and print the answer along with the sources.
    result = answer(args.question, results, make_claude_llm())
    print(f"Answer:\n{result['text']}\n")
    print("Sources:")
    for i, source in enumerate(result["sources"], start=1):
        print(f"  [{i}] {source}")


if __name__ == "__main__":
    main()
