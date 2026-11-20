# Codebase Guide

A living reference for every file in this project: what each file does, and what each function inside it does. It is updated as the project grows.

> **Why this file is not in `docs/`:** `docs/` is the RAG corpus. `build_index` indexes every `*.md` file in that folder, so a code guide placed there would be embedded and retrieved as if it were a company policy.

## Pipeline at a glance

```
docs/*.md
  -> parse_frontmatter   (YAML metadata: policy_id, version, status, ...)
  -> split_sections      (one piece per markdown heading)
  -> chunk_text          (180-word windows, 30-word overlap)
  -> add_chunks          (embed with all-MiniLM-L6-v2, upsert into chromadb)
  -> search              (embed the question, return top-k closest chunks)
```

| Status | Part |
|---|---|
| Done | Chunking, embedding and storage, ingest, retrieval CLI |
| Done | LLM answer generation with sources (`rag/answer.py`) |
| Done | Hybrid search (`rag/hybrid.py`) |
| In progress | Reranking (`rag/rerank.py`) |
| Later | Eval harness, diagnosis write-up, CI |

---

## `rag/chunking.py`

Turns one markdown policy document into a list of chunk dictionaries ready for embedding. It splits by section first, then by word window, and copies the document's frontmatter onto every chunk. Pure Python with no dependencies, so it is fast to test.

### `parse_frontmatter(markdown)`

Splits a document into its YAML-style frontmatter and its body.

- **Input:** the full markdown string.
- **Returns:** `(meta, body)`. `meta` is a dict such as `{"policy_id": "FIN-204", "version": "1.4", "status": "superseded"}`.
- Only runs if the text starts with `---`. Otherwise it returns `({}, markdown)` unchanged.
- Each `key: value` line is split on the first colon, so values that contain colons survive. Surrounding quotes are stripped (`"1.4"` becomes `1.4`).
- Lines without a colon are ignored. This is a minimal parser, not full YAML: nested lists and multi-line values are not supported.

### `split_sections(markdown)`

Splits a markdown body into sections, one per heading.

- **Returns:** a list of `{"section": <heading text>, "text": <body text>}`.
- A line is a heading if it starts with one or more `#` characters followed by whitespace or the end of the line. So `#hashtag` is not a heading.
- Text before the first heading goes into a section called `"Preamble"`.
- Headings with no body text are skipped, via `_append_if_content`.
- Closing hashes are removed: `## Meals ##` becomes `Meals`.
- The section name is later used as the source-attribution label.

### `_finish_section(current)` (private)

Converts the in-progress section, `{"section", "lines"}`, into its final form, `{"section", "text"}`. It joins the lines with newlines and strips the result. It is a pure function and does not mutate its argument.

### `_append_if_content(sections, current)` (private)

Appends the finished section to `sections` only if its text is not blank. Both places in `split_sections` that close a section (on a new heading, and at end of file) share this function.

### `chunk_text(text, size, overlap)`

Splits text into overlapping windows of words.

- **Returns:** a list of strings, each at most `size` words long.
- Each window starts `size - overlap` words after the previous one, so neighbouring chunks share `overlap` words.
- It stops as soon as a window reaches the last word, so it never produces a redundant tail chunk.
- Raises `ValueError` if `overlap >= size`. Without this check the start index would never advance, and the loop would run forever.
- It counts words, not characters, so no word is ever cut in half.

### `chunk_document(markdown, source, size=180, overlap=30)`

The public entry point. It runs `parse_frontmatter`, then `split_sections`, then `chunk_text`.

- **Returns:** a list of chunk dicts. Each one contains:
  - every frontmatter field;
  - `id`: `"<source>::<section>::<index>"`, which is unique and stable across runs, so upserts replace the old chunk instead of duplicating it;
  - `chunk`: the text;
  - `section`;
  - `source`: the file name.
- Frontmatter is unpacked first, so the pipeline's own keys win if a frontmatter key has the same name.
- `size=180` stays under all-MiniLM-L6-v2's 256-token limit. Text past that limit would be cut off before embedding.
- `overlap=30` (about 15%) keeps a sentence that crosses a window boundary whole in at least one chunk.
- On the current corpus every section fits in one chunk; the largest is about 165 words.

---

## `rag/store.py`

The embedding and vector-store layer, a thin wrapper around sentence-transformers and chromadb. The embedding function is passed in as a parameter, so tests can use a fast fake embedder instead of loading the real model.

### `make_embedder(model_name)`

Loads a `SentenceTransformer` model once and returns a closure, `embed_function(texts)`.

- Call it once per process. Loading the model is the slow part.
- In production `model_name` is `"all-MiniLM-L6-v2"`: 384 dimensions, maximum 256 tokens.

#### `embed_function(texts)` (inner closure)

Encodes a list of strings and returns plain Python lists of floats, which is what chromadb accepts.

### `add_chunks(collection, chunks, embed_function)`

Embeds chunks and stores them in a chromadb collection.

- Returns immediately for an empty list. chromadb raises `ValueError` on an empty write.
- The chunk text is stored as the document. Every other field except `id` and `chunk` is stored as metadata, so the text is not stored twice.
- Uses `upsert`, not `add`. Re-running ingest replaces existing ids instead of silently ignoring them.
- Metadata values must be `str`, `int`, `float` or `bool`. chromadb rejects `None`, lists and dicts.

### `search(collection, query, embed_function, k=5)`

Embeds the query and returns the `k` nearest chunks, closest first.

- **Returns:** a list of dicts. Each one contains every stored metadata field (`section`, `source`, `policy_id`, `version`, `status`, ...), plus:
  - `id`;
  - `chunk`;
  - `distance`: a float, where lower means closer.
- Metadata is unpacked first, so `id`, `chunk` and `distance` cannot be overwritten by a stored field.
- chromadb caps `k` at the collection size.

---

## `rag/ingest.py`

Connects chunking to storage: it builds the whole index from a folder of documents.

### `build_index(folder, collection, embedder)`

Chunks every `*.md` file in `folder` and stores all the chunks in one `add_chunks` call.

- **Returns:** the number of chunks indexed. This is 20 for the current `docs/`.
- `folder` can be a `str` or a `Path`.
- Files are processed in sorted order, so results are the same on every machine.
- Only matches `*.md` and does not search subfolders.
- Running it twice is safe and adds no duplicates, because `add_chunks` upserts.

---

## `rag/answer.py` (in progress, `feature/answer`)

Turns retrieved chunks and a question into a grounded LLM answer with sources.

Module constants:

- `MODEL`: `"claude-opus-5-5"`.
- `SYSTEM_PROMPT`: the rules Claude follows for every answer:
  - answer only from the numbered chunks;
  - cite inline as `[n]`;
  - if the answer isn't there, say "I couldn't find that in the policy documents.";
  - when versions conflict, answer from the `active` chunk and mention the `superseded` figure;
  - keep answers short.

### `build_prompt(question, results)`

Builds the prompt text sent to the LLM.

- **Input:** the question, and the result dicts returned by `search`.
- **Returns:** a string. It starts with `Question: ...`, followed by each chunk as:
  ```
  [n] Source: <file>, Section: <section>, Version: <v>, Status: <status>
  <chunk text>
  ```
- The `[n]` labels let the model cite its sources inline.
- `Status` lets the model prefer `active` chunks over `superseded` ones. This is the hook for the planted expense-policy conflict.
- Missing metadata falls back to `unknown`, `?` or `n/a` instead of raising `KeyError`.

### `answer(question, results, llm_fn)`

Produces the final answer.

- **Returns:** `{"text", "prompt", "sources"}`.
- `llm_fn` is passed in as a parameter: any function that takes a prompt string and returns text. Tests pass a fake, so they never call the API.
- If `results` is empty, it returns immediately with a "Couldn't find..." message, `prompt: None` and `sources: []`. `llm_fn` is never called, which saves an API call and stops the model answering from general knowledge.
- `sources` is a list of `"source > section"` strings in rank order. They are built from the retrieved metadata, not from the LLM's output, so they are always accurate (Part 7, attribution).

### `make_claude_llm(model=MODEL)`

Creates the real `llm_fn`.

- Creates one `anthropic.Anthropic()` client, which reads `ANTHROPIC_API_KEY` from the environment, and returns a closure that reuses it. This is the same factory pattern as `make_embedder`.
- It has no unit test, because it is a thin wrapper around the API. A one-line smoke test checks it, and Part 5's eval covers it end to end.

#### `llm_fn(prompt)` (inner closure)

Calls `client.messages.create` with `SYSTEM_PROMPT` and `max_tokens=1024`.

- If `stop_reason == "refusal"`, it returns a short fallback message.
- Otherwise it joins the `.text` of every block whose `type == "text"`. SDK blocks are objects, not dicts, so it uses attribute access.

---

## `rag/hybrid.py` (`feature/hybrid`)

Hybrid retrieval: a hand-built BM25 keyword search, fused with vector search using Reciprocal Rank Fusion. Embeddings blur exact identifiers and rare terms. Keyword search matches them exactly but misses paraphrases. Fusing the two gets the strengths of both.

Module constants:

- `TOKEN_PATTERN`: runs of `[a-z0-9]`, optionally joined by hyphens, so `FIN-204` stays one token.
- `BM25_K1 = 1.5`: term-frequency saturation.
- `BM25_B = 0.75`: length normalisation.

### `tokenize(text)`

Lowercases the text and returns its word tokens with punctuation removed. `"$70,"` becomes `"70"` and `"FIN-204."` becomes `"fin-204"`.

### `keyword_search(chunks, query, k=5)`

Scores each chunk against the query with BM25.

- **Returns:** `[{**chunk, "score"}]`, best first.
- Chunks with no matching term are left out.
- **IDF:** rare terms such as `fin-204` outweigh common ones such as `policy`.
- **Saturating TF:** repeating a word helps less each time.
- **Length normalisation:** long chunks don't win just by containing more words.
- An empty `chunks` list returns `[]`.

### `rrf_fuse(rankings, k=60)`

Merges ranked lists. Each item scores `sum(1 / (k + rank))` over the lists it appears in.

- **Returns:** a list deduplicated by `id`, keeping the first-seen copy of each item plus its `rrf_score`.
- It works on ranks, not raw scores, so BM25 scores and cosine distances never have to share a scale.
- `k` is the smoothing constant (60 is the standard value). It is not the number of results.

### `_all_chunks(collection)` (private)

Reads every stored chunk back out of chromadb as `{**metadata, "id", "chunk"}`. BM25 needs the whole corpus to compute document frequencies.

### `hybrid_search(collection, query, embed_function, k=5, candidates=20)`

Takes `candidates` results from vector search and from keyword search, fuses them, and returns the top `k`.

- `candidates` is larger than `k` on purpose, so a chunk ranked low by one retriever but high by the other can still win.
- Vector results are passed first, so their copy, which carries `distance`, is the one kept.
- **Demo query (real model):** "Can I get reimbursed for extra peripherals?"
  - Vector-only top 3 are all expense chunks, which miss the answer.
  - Hybrid lifts `remote-work-policy.md > Equipment and information security` to #2, through an exact match on "peripherals".

---

## `rag/rerank.py` (`feature/rerank`)

Second-stage reranking with a CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). A bi-encoder embeds the query and each chunk separately and compares the vectors. A cross-encoder reads the query and chunk together in one pass, so it judges relevance more precisely, but it is slower. It is therefore run only on the top ~10 candidates from hybrid search.

### `make_reranker(model_name)`

Loads the `CrossEncoder` once and returns a closure, `scorer(pairs)`. This is the same factory pattern as `make_embedder` and `make_claude_llm`.

#### `scorer(pairs)` (inner closure)

Takes a list of `(query, chunk)` tuples and returns one float per pair; higher means more relevant. It converts numpy values to plain Python floats.

### `rerank(query, results, scorer, top_n=3)`

Re-scores the retrieved results and returns the top `top_n` as `[{**r, "rerank_score"}]`, best first.

- With empty `results`, it returns `[]` without calling the scorer.
- It calls the scorer once with every pair, in a single batch.
- It builds new dicts instead of modifying the input, so the caller can still print the original order for a before/after comparison.
- **Demo:** for the peripherals query, the Equipment chunk goes from #2 after hybrid to #1 after rerank.

---

## `rag/ask.py`

The command-line entry point. It rebuilds the index, retrieves the chunks for a question (vector or hybrid, optionally reranked), prints them, and then prints Claude's cited answer and the sources.

```bash
HF_HUB_OFFLINE=1 python -m rag.ask "Can I get reimbursed for extra peripherals?" --mode hybrid --rerank --k 3
```

Module constants:
- `PROJECT_ROOT`, `DOCS_DIR`, `DB_DIR`: paths built from the file's own location, so the script works from any working directory.
- `MODEL_NAME`: `"all-MiniLM-L6-v2"`.
- `RERANK_MODEL`: `"cross-encoder/ms-marco-MiniLM-L-6-v2"`.
- `RERANK_CANDIDATES`: 10, the size of the pool retrieved before reranking.
- `COLLECTION_NAME`: `"policies"`.

### `get_collection()`

Opens the on-disk chromadb collection in `chroma_db/`, creating it if needed. `chroma_db/` is listed in `.gitignore`.

### `score_label(r)`

Returns the score that decided a result's rank, checking the most specific first: `rerank=`, then `rrf=`, then `distance=`. A keyword-only hybrid hit has no distance, so this avoids a `KeyError`.

### `print_results(results, preview=True)`

Prints each result as `[rank] <score> source > section (vVERSION, STATUS)`. With `preview=True` it adds a 200-character preview of the chunk. The compact form is used for the "Before rerank" list.

### `main()`

1. Parses the arguments:
   - `question`;
   - `--k`, which defaults to 3;
   - `--mode vector|hybrid`, which defaults to `hybrid`;
   - `--rerank`.
2. Loads the embedder once.
3. Re-indexes `docs/`. This takes about a second for 20 chunks.
4. Retrieves the chunks with `search` or `hybrid_search`: `k` results, or `RERANK_CANDIDATES` when `--rerank` is set.
5. Prints the mode and the question.
6. With `--rerank`, it prints the candidates compactly under "Before rerank", reranks them down to `k`, then prints "After rerank".
7. Prints the final results with previews.
8. Calls `answer(..., make_claude_llm())` and prints `Answer:` and the numbered `Sources:`.

---

## `rag/eval_set.py` (`feature/eval`)

`EVAL_SET` is the Part 5 question set: 10 questions. Each entry has:

- `question`;
- `expected_source` and `expected_section`, the chunk that should be retrieved;
- `key_facts`: short exact strings that a correct answer must contain, such as `"$70"`, `"30"` or `"31 March"`.

The set covers all 4 documents, all 3 conflicting expense facts (which expect the **v2.0** source), and one question the documents don't answer ("gym membership"). That question has `expected_source: None` and must produce "couldn't find".

Key facts are short substrings, not sentences, because Claude rewords its answers.

---

## `example_loop/minimal_loop.py`

Part 2's minimal loop: a standalone script that proves embed, store and retrieve work before any pipeline code exists. It embeds 3 sentences into an in-memory chromadb collection and asks "How many days can I work from home?". It then asserts that the remote-work sentence ranks first and prints `PASS`. It has no functions; it is a top-to-bottom script.

---

## `docs/` (the corpus)

Four policy documents, each with YAML frontmatter.

| File | Policy | Version | Status |
|---|---|---|---|
| `expense-reimbursement-policy-v1.4.md` | FIN-204 | 1.4 | superseded (1 Jan to 30 Sep 2026) |
| `expense-reimbursement-policy-v2.0.md` | FIN-204 | 2.0 | active (from 1 Oct 2026) |
| `paid-time-off-policy.md` | HR-305 | | active |
| `remote-work-policy.md` | HR-101 | 1.2 | active |

**Planted data quality issue:** both expense versions are indexed, and they conflict.

| Rule | v1.4 | v2.0 |
|---|---|---|
| Daily meal cap | $55 | $70 |
| Claim deadline | 45 days | 30 days |
| Itemised receipt threshold | $50 | $25 |

---

## Tests (`tests/`)

Run the fast suite with `python -m pytest -m "not slow"`. Run everything with `HF_HUB_OFFLINE=1 python -m pytest`; the `slow` tests load the real model.

### `tests/conftest.py`

Shared fixtures. pytest finds this file automatically, and a test gets a fixture by naming it as a parameter.

- `_fake_embed(texts)`: returns a 3-number vector per text: `[count of "cap", count of "economy", 1.0]`. Its distances are predictable, so tests can check the ranking. The constant `1.0` keeps every vector non-zero.
- `fake_embed` (fixture): returns `_fake_embed`.
- `collection` (fixture): a fresh, empty chromadb collection with a uuid name. `EphemeralClient` instances share one store per process, so a fixed name would leak data between tests.

### `tests/test_chunking.py` (19 tests)

- **split_sections:**
  - finds section names;
  - keeps the body text;
  - keeps text before the first heading as the Preamble;
  - skips headings with no body.
- **parse_frontmatter:**
  - reads `policy_id`;
  - strips quotes;
  - returns the body after the closing fence;
  - returns input without frontmatter unchanged.
- **chunk_text:**
  - short text gives one chunk;
  - no extra chunk at the end;
  - rejects overlap >= size;
  - no words are lost;
  - no chunk exceeds the size;
  - neighbouring chunks share the overlap.
- **chunk_document:** attaches the section, source, `policy_id` and version; ids are unique.

### `tests/test_store.py` (15 tests)

- `_stored_metadata(collection, chunk_id)` (helper): reads back the metadata stored for one id.
- **add_chunks:**
  - stores every chunk;
  - stores the text as the document;
  - keeps section metadata;
  - keeps frontmatter metadata;
  - does not copy the text into metadata;
  - an empty list stores nothing;
  - a second call updates the chunk instead of ignoring it.
- **search:**
  - returns `k` results;
  - ranks the closest chunk first;
  - includes the chunk text;
  - includes the section;
  - includes a float distance;
  - `k` larger than the collection returns everything;
  - includes frontmatter metadata.
- **slow:** the real model matches the "food" question to the Meals chunk.

### `tests/test_ingest.py` (8 tests)

- `DOCS_DIR`: the path to the real `docs/` folder.
- `_write_two_docs(folder)` (helper): writes `a.md` and `b.md` into a temp folder.
- **build_index:**
  - stores chunks from every document;
  - returns the chunk count;
  - ignores non-`.md` files;
  - an empty folder returns 0;
  - carries frontmatter metadata;
  - running twice adds no duplicates;
  - accepts a `str` path.
- **slow:** on the real corpus, the work-from-home question retrieves `remote-work-policy.md` first.

### `tests/test_answer.py` (7 tests)

- `CHUNKS`: two fake search results (Meals, Travel) from `a.md`.
- `_fake_llm(prompt)`: returns a canned answer.
- `_exploding_llm(prompt)`: raises if called. It proves the LLM was skipped.
- **build_prompt:**
  - includes the question;
  - includes the chunk text;
  - labels each chunk as `[n] Source: ..., Section: ...`.
- **answer:**
  - returns the LLM text;
  - lists sources in rank order;
  - with no results, does not call the LLM;
  - with no results, says "couldn't find".

### `tests/test_hybrid.py` (23 tests, 2 slow, `feature/hybrid`)

- **tokenize:**
  - lowercases;
  - strips punctuation;
  - keeps hyphenated IDs such as `fin-204` whole.
- **keyword_search** (on `KW_CHUNKS` and `IDF_CHUNKS`):
  - an exact term match ranks first;
  - chunks with no matching term are excluded;
  - no match returns an empty list;
  - returns at most `k` results;
  - each result has a float score;
  - other chunk fields are kept;
  - a rare term outranks a repeated common term (IDF).
- **rrf_fuse:**
  - an item first in both lists ranks first;
  - an item in both lists beats an item at the top of only one;
  - no duplicates;
  - the score is `1/(k+rank)`;
  - item fields are kept;
  - an empty input returns an empty list.
- **hybrid_search:**
  - **sanity check:** vector-only misses the exact-ID chunk. The fake embedder counts substrings, so it sees "caps" as "cap";
  - hybrid ranks the exact-ID chunk first;
  - returns `k` results;
  - keeps the metadata;
  - includes the chunk text.
- **slow** (real corpus, "Can I get reimbursed for extra peripherals?"):
  - vector-only top 3 misses the Equipment chunk;
  - hybrid top 3 includes it.

### `tests/test_rerank.py` (13 tests, 3 slow, `feature/rerank`)

- `RESULTS`: three fake candidates.
- `_fake_scorer`: scores each chunk by how many times it says "peripherals".
- `_exploding_scorer`: raises if called.
- **rerank:**
  - the highest-scored chunk comes first;
  - all results are ordered by score;
  - returns at most `top_n`;
  - a `top_n` larger than the input returns everything;
  - each result has a float `rerank_score`;
  - original fields are kept;
  - the scorer receives `(query, chunk)` pairs;
  - the input list is not modified;
  - empty input does not call the scorer;
  - empty input returns an empty list.
- **slow** (`cross-encoder/ms-marco-MiniLM-L-6-v2`):
  - `make_reranker` returns one float per pair;
  - a relevant chunk scores higher than an irrelevant one;
  - on the real corpus, hybrid top 10 followed by rerank puts `Equipment and information security` at #1 for the peripherals query.

### `tests/test_eval.py` (34 tests, `feature/eval`)

The Part 5 eval harness. It uses the full pipeline: hybrid top 10, then rerank top 3, then Claude.

- **Fixtures** (module scope, so they are built once):
  - `retrieve(question)` indexes the real documents into an ephemeral collection and runs hybrid search plus reranking.
  - `answer_text(question)` caches Claude's answer for each question, so a question with 2 key facts still makes only 1 API call.
- **Fast checks of the eval set:**
  - it has at least 8 questions;
  - it includes an unanswerable question;
  - every expected section exists in the corpus, which catches typos.
- **`test_retrieval_recall_at_3`** (slow, 9 cases): the expected `(source, section)` is in the top 3. The result is **9/9**. For comparison, hybrid alone gets 8/9 and vector alone 7/9.
- **`test_superseded_policy_is_not_retrieved`** (slow, 3 cases, `xfail(strict=True)`): records the planted issue. Superseded v1.4 still appears in the top 3 for every expense question, and ranks #1 for the meal and deadline questions even after reranking. `strict=True` turns a pass into a test failure, so fixing the corpus forces this marker to be removed.
- **`test_answer_contains_key_fact`** (slow + `eval`, 11 cases): each key fact appears in the answer. Curly apostrophes are normalised first. These tests are skipped when `ANTHROPIC_API_KEY` isn't set.

---

## Secrets

`ANTHROPIC_API_KEY` is set in `~/.zshrc`, never in a file inside the repo. For CI it goes in the GitHub repo's Secrets.

---

## Config files

- `pyproject.toml`: pytest settings:
  - `pythonpath = ["."]`, so `import rag` works;
  - `testpaths`;
  - the `slow` marker (real models);
  - the `eval` marker (paid Claude API calls);
  - a filter that hides chromadb's `asyncio` deprecation warning.
- `.gitignore`: `__pycache__/`, `.venv/`, `*.pyc`, `chroma_db/`.
