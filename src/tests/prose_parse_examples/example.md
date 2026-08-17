# mnrva - Development & Architecture Document

## 1. Overview

A retrieval-augmented system for querying a code repository in natural language
(e.g. "how is auth implemented?"). The system indexes a codebase into
structure-aware, context-enriched chunks, answers developer queries through an
agent with multiple retrieval tools, and keeps the index in sync as the
codebase changes.

Primary goal: structure-aware chunking, contextual retrieval, and
agentic (multi-tool) retrieval - applied to the source code domain.

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
