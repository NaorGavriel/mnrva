# Code Repository RAG — Development & Architecture Document

## 1. Overview

A retrieval-augmented system for querying a code repository in natural language
(e.g. "how is auth implemented?"). The system indexes a codebase into
structure-aware, context-enriched chunks, answers developer queries through an
agent with multiple retrieval tools, and keeps the index in sync as the
codebase changes.

Primary goal: structure-aware chunking, contextual retrieval, and
agentic (multi-tool) retrieval — applied to the source code domain.

## 2. System Components

The system is organized into three components.

### 2.1 Ingestion & Indexing Pipeline

Responsible for turning raw repository files into searchable, enriched chunks.

For each source file:
1. Route to the correct tree-sitter grammar by file extension (see §3.3).
2. Parse the file and extract function/class/method-level nodes using a
   per-language node-type query.
3. Extract imports from the same parse tree (per-file)
4. Generate a short contextual description of each chunk using an LLM, given
   the whole file and its imports (prompt-cached to control cost).
5. Prepend the context to the raw chunk, embed it, and upsert into the
   database with a deterministic ID and metadata (`file_path`,
   `symbol_name`, `start_line`, `end_line`, `language`, `imports`).

Chunk IDs are derived from `file_path + symbol_name` but formatted as a
UUID (via `uuid5`). Qdrant only accepts 64-bit unsigned integers or UUIDs as point IDs. The mapping stays deterministic which is what lets the refresh pipeline (§2.3) do a clean delete-and-replace on re-index.

Non-code files (README, config files, docs) follow a separate path: chunked
by semantic-similarity.

### 2.2 Query Agent

Handles developer-facing natural language queries against the index.

Given a query, the agent chooses between retrieval tools:

- **Hybrid search** — dense vector similarity + BM25 keyword search over
  chunks, combined server-side by Qdrant's Query API (`prefetch` +
  `FusionQuery(fusion=RRF)`). Vector search catches conceptual matches; BM25
  catches exact identifier/error-string matches.
  Reciprocal Rank Fusion merges the two ranked lists by rank positions. Native to Qdrant.
- **Whole-file read** — for queries that need full context a chunk can't
  provide (e.g. "how is auth implemented end to end").
- **Symbol / grep search** — for exact lookups (e.g. "where is `retry_with_backoff` defined").

Cross-file context is handled by the agent's own tool loop — reading a
file, noticing an import, and opening the imported file directly.

### 2.3 Refresh & Sync Pipeline

Keeps the index consistent as the codebase changes. A deterministic sync job.

For each file in the diff since the last indexed commit:
- **Deleted file** — remove all chunks where `file_path` matches.
- **Changed/added file** — delete existing chunks for that `file_path`,
  re-parse, re-chunk, re-enrich, re-embed, and re-insert.

Because chunk IDs are deterministic (file path + symbol name), this is a
straightforward delete-and-replace.

Rename detection is out of scope for MVP. Renames are treated as delete+add.

## 3. System Architecture

### 3.1 Language

Python.

### 3.2 Libraries & Frameworks

- **LangChain** — repo/document loaders, agent/tool orchestration.
- **tree-sitter** — AST-aware chunking. One grammar package per supported
  language (see §3.3).

### 3.3 Multi-language support

The codebase is not assumed to be single-language. Routing and extraction
are language-specific:

| File extension | Grammar package | Notes |
|---|---|---|
| `.py` | `tree_sitter_python` | `function_definition`, `class_definition` |
| `.ts`, `.tsx` | `tree_sitter_typescript` | `function_declaration`, `class_declaration`, `interface_declaration`, plus arrow functions assigned via `variable_declarator` |
| `.js`, `.jsx` | `tree_sitter_javascript` | same arrow-function caveat as TypeScript |

Each language has its own node-type query, maintained as a small per-language config.

A `language` field is stored in chunk metadata, enabling language-scoped
filtering or boosting at query time.

### 3.4 Database

**Qdrant.** A single store covers every data need in MVP scope.

- **Vectors** — dense embeddings for semantic search.
- **Sparse vectors** — Qdrant generates BM25 sparse vectors natively for the lexical half of hybrid search.
- **Payload** — chunk metadata (`file_path`, `symbol_name`, `start_line`,
  `end_line`, `language`, `imports`) stored per point, with indexed fields
  for fast filtering (e.g. language-scoped search).
- **Fusion** — dense + sparse results are combined server-side via the
  Query API's built-in RRF, no custom fusion code required (§2.2).

**Practical constraint:** Qdrant point IDs must be 64-bit unsigned integers
or UUIDs.

## 4. Low-Level Plan — Component 1: Ingestion & Indexing Pipeline

### 4.1 File structure

```
models.py
languages.py
repository_clone.py
repository_parser.py
tree_parser.py
prose_parser.py
context_enricher.py
embeddings.py
db.py
repository_ingester.py
```

Scope note: this pipeline assumes ingestion always begins with
`repository_clone.py`, so `.git` is guaranteed present by the time
`repository_parser.py` runs.

### 4.2 `models.py`

Shared data contract used by every other module in this component (and
reused by the Refresh & Sync Pipeline, §2.3).

- `Chunk` — dataclass: `id`, `file_path`, `symbol_name`, `start_line`,
  `end_line`, `language`, `imports: list[str]`, `raw_text`,
  `context_text: str | None`, `embedding: list[float] | None`.
- `generate_chunk_id(file_path: str, symbol_name: str) -> str` —
  deterministic `uuid5`-based ID (see §3.4 for why point IDs must be
  UUID-formatted). Lives here rather than in `db.py` since it's part of how
  a `Chunk` is constructed, and both `tree_parser.py` and `prose_parser.py`
  need it.

### 4.3 `languages.py`

Single source of truth for language/extension configuration — avoids the
extension list existing separately in the ingestion allowlist and the
tree-sitter routing table.

- `LANGUAGE_CONFIG` — extension → `{grammar_package, node_types}`, per §3.3.
- `get_language(path: Path) -> str | None`
- `is_code_file(path: Path) -> bool`
- `is_prose_file(path: Path) -> bool`

### 4.4 `repository_clone.py`

- `clone_repository(github_url: str, dest_dir: Path) -> Path`
- `get_current_commit_sha(repo_path: Path) -> str` — needed by the Refresh
  & Sync Pipeline to compute its diff.

### 4.5 `repository_parser.py`

Returns the filtered file list for ingestion:

- `list_source_files(repo_path: Path) -> list[Path]` — `git ls-files`,
  filtered through the extension allowlist (`languages.py`), a filename
  denylist (e.g. `package-lock.json`), and a file-size cutoff.
- `_passes_size_cutoff(path: Path, max_bytes: int) -> bool`
- `_is_denied_filename(path: Path) -> bool`

### 4.6 `tree_parser.py`

- `parse_code_file(path: Path, language: str) -> list[Chunk]` — loads the
  grammar via `languages.py`, runs the per-language node-type query, slices
  source by byte offsets into `Chunk` objects (`raw_text` and `id`
  populated; `context_text`/`embedding` left `None`).
- `_extract_imports(tree, source: bytes, language: str) -> list[str]`
- `_load_grammar(language: str)` — cached per language.

### 4.7 `prose_parser.py`

Handles non-code files (`.md`, `.txt`, config files) via semantic-similarity.

- `parse_prose_file(path: Path) -> list[Chunk]` — same `Chunk` shape as the
  code path; `symbol_name` derived from section heading or chunk index.

### 4.8 `context_enricher.py`

Split out as its own module rather than folded into general utilities,
since contextual retrieval is a core technique this project demonstrates,
not an incidental helper.

- `enrich_chunk(chunk: Chunk, full_file_text: str) -> str` — the LLM call;
  full file content is prompt-cached so only the chunk-specific instruction
  varies per call.
- `_build_enrichment_prompt(chunk: Chunk, full_file_text: str) -> str`

### 4.9 `embeddings.py`

- `embed_text(text: str) -> list[float]` — dense embedding only. Sparse
  (BM25) vectors are generated by Qdrant itself at upsert time (§5.10), so
  no hand-written sparse embedding function is needed here.

### 4.10 `db.py`

- `init_client(path: str | None = None, url: str | None = None) -> QdrantClient`
  — one function, two ways to call it (local file path during development,
  connection string/URL in CI and production). This is what makes moving
  from local mode to a hosted instance a one-line change rather than a
  rewrite (§5.12).
- `ensure_collection(client, name: str) -> None`
- `upsert_chunk(client, chunk: Chunk) -> None` — writes dense embedding,
  raw text (for Qdrant's native BM25 sparse generation), and payload
  metadata.
- `delete_by_file(client, file_path: str) -> None`
- `hybrid_search(client, query: str, language: str | None = None, limit: int = 10) -> list[SearchResult]`
  — `prefetch` + `FusionQuery(fusion=RRF)`, per §2.2.
- `get_last_indexed_commit(client) -> str | None` /
  `set_last_indexed_commit(client, sha: str) -> None` — stored as a small
  dedicated metadata point in the same collection; this is where the
  Refresh & Sync Pipeline's "last indexed commit" state (§2.3) actually
  lives.

### 4.11 `repository_ingester.py`

- `ingest_file(path: Path, client) -> None` — routes to `tree_parser` or
  `prose_parser` via `languages.is_code_file`, enriches, embeds, upserts.
  Built as a standalone per-file unit specifically because the Refresh &
  Sync Pipeline (§2.3) calls this exact function per changed file rather
  than duplicating its logic.
- `ingest_repository(repo_path: Path, client) -> None` — lists files via
  `repository_parser`, calls `ingest_file` for each, then
  `set_last_indexed_commit`.