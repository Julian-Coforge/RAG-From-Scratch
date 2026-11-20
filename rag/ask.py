import argparse
from pathlib import Path

import chromadb

from rag.ingest import build_index
from rag.store import make_embedder, search
from rag.answer import answer, make_claude_llm
from rag.hybrid import hybrid_search
from rag.rerank import make_reranker, rerank


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DB_DIR = PROJECT_ROOT / "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 10
COLLECTION_NAME = "policies"


def get_collection():
    """Open (or create) the on-disk chromadb collection in chroma_db/."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def score_label(r):
    """Show the score that actually decided this result's rank."""
    # Keyword-only hybrid hits have no distance, so check the most specific score first
    if "rerank_score" in r:
        return f"rerank={r['rerank_score']:.4f}"
    if "rrf_score" in r:
        return f"rrf={r['rrf_score']:.4f}"
    return f"distance={r['distance']:.4f}"


def print_results(results, preview=True):
    """Print ranked results with score, source, section, version, status and an optional preview."""
    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] {score_label(r)}  {r['source']} > {r['section']}"
              f"  (v{r.get('version', '?')}, {r.get('status', 'n/a')})")
        if preview:
            print(f"    {r['chunk'][:200]}{'...' if len(r['chunk']) > 200 else ''}\n")


def main():
    parser = argparse.ArgumentParser(description="Ask a question over the policy documents.")
    parser.add_argument("question", help="natural-language question")
    parser.add_argument("--k", type=int, default=3, help="number of chunks to retrieve (default 3)")
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid",
                    help="retrieval method (default hybrid)")
    parser.add_argument("--rerank", action="store_true",
                        help=f"retrieve {RERANK_CANDIDATES} candidates and rerank them with a CrossEncoder")
    args = parser.parse_args()

    # Load the embedding model once and reuse it for both indexing and the query
    embed = make_embedder(MODEL_NAME)

    # Build the index (if not already built) and open the collection
    collection = get_collection()
    count = build_index(DOCS_DIR, collection, embed)
    print(f"Indexed {count} chunks from {DOCS_DIR.name}/ into {DB_DIR.name}/ (collection size: {collection.count()})")

    # Reranking needs a wider candidate pool than the final k
    n_retrieve = RERANK_CANDIDATES if args.rerank else args.k

    # Search for the most relevant chunks based on the question and the specified mode
    if args.mode == "hybrid":
        results = hybrid_search(collection, args.question, embed, k=n_retrieve)
    else:
        results = search(collection, args.question, embed, k=n_retrieve)

    print(f"\nMode: {args.mode}{' + rerank' if args.rerank else ''}\nQuestion: {args.question}\n")

    if args.rerank:
        print(f"Before rerank (top {n_retrieve} from {args.mode}):")
        print_results(results, preview=False)
        results = rerank(args.question, results, make_reranker(RERANK_MODEL), top_n=args.k)
        print(f"\nAfter rerank (top {args.k}):")

    print_results(results)

    # Ask the LLM to answer the question using the retrieved chunks and print the answer along with the sources.
    result = answer(args.question, results, make_claude_llm())
    print(f"Answer:\n{result['text']}\n")
    print("Sources:")
    for i, source in enumerate(result["sources"], start=1):
        print(f"  [{i}] {source}")


if __name__ == "__main__":
    main()
