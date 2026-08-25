# mnrva - Development & Architecture Document

## 1. Overview

A retrieval-augmented system for querying a code repository in natural language
(e.g. "how is auth implemented?"). The system indexes a codebase into
structure-aware, context-enriched chunks, answers developer queries through a
corrective-RAG agent, and keeps the index in sync as the codebase changes.

Primary goal: structure-aware chunking, contextual retrieval, and
corrective (retrieve/grade/generate/evaluate) retrieval - applied to the
source code domain.

## 2. System Components

The system is organized into three components.

### 2.1 Ingestion & Indexing Pipeline

Responsible for turning raw repository files into searchable, enriched chunks.
Runs as a producer/consumer pipeline so throughput scales with repo size under a shared, account-level rate budget.

1. **Parse phase.** Walk every wanted file (git-tracked, extension-allowlisted,
   filename-denylisted). Route each to the correct tree-sitter grammar by
   file extension (see §3.3) and extract function/class/method-level nodes
   plus imports via a per-language node-type query; non-code files are
   chunked by prose/semantic splitting instead. No LLM calls in this phase.
   All parsed files feed a repo-wide file queue.
2. **Enrichment workers.** A bounded pool of workers pulls files off that
   queue. Each worker enriches a file's chunks in one batched,
   structured-output LLM call (the file source/imports sent once, covering
   all of that file's chunks, to avoid re-sending the file per chunk);
   large files are split into sequential sub-batch calls. 
   Calls are paced through a shared async token-bucket
   rate limiter (requests/minute and tokens/minute), sized from the
   account's actual limits rather than hardcoded. A file already durably
   upserted from a prior run is skipped.
   Enriched chunks are pushed onto a second, ready-to-embed queue.
3. **Embedding consumer.** Drains the ready-to-embed queue, accumulating
   chunks *across files* until a full embedding-batch is available (or the
   enrichment side signals it's fully drained),
   then embeds and upserts the whole batch in one call each. Embedding calls share the same rate limiter as enrichment, since both draw from the same account-level budget.

A batch becoming durable in Qdrant is the pipeline's recovery unit: a crash
only loses whatever hadn't yet been embedded/upserted, and re-parsing on
restart is cheap since it's local and LLM-free.

Chunk IDs are derived from `file_path + symbol_name` but formatted as a
UUID (via `uuid5`). Qdrant only accepts 64-bit unsigned integers or UUIDs as point IDs. The mapping stays deterministic which is what lets the refresh pipeline (§2.3) do a clean delete-and-replace on re-index.

Non-code files (README, config files, docs) follow a separate path: chunked
by semantic-similarity.

Ingestion also seeds the repo's `github_url`/`commit_sha` into Postgres
(§3.5) — the record the refresh pipeline updates and the query agent reads.

### 2.2 Query Agent

Handles developer-facing natural language queries against the index.

Given a query, the agent retrieves via:

- **Hybrid search** — dense vector similarity + BM25 keyword search over
  chunks, combined server-side by Qdrant's Query API (`prefetch` +
  `FusionQuery(fusion=RRF)`). Vector search catches conceptual matches; BM25
  catches exact identifier/error-string matches.
  Reciprocal Rank Fusion merges the two ranked lists by rank positions. Native to Qdrant.



Conversation state persists in Postgres (§3.5) via LangGraph's checkpointer,
keyed by a per-conversation `thread_id`. Full design: `docs/query_agent.md`.

### 2.3 Refresh & Sync Pipeline

Keeps the index consistent as the codebase changes. A deterministic sync job.

For each file in the diff since the last indexed commit:
- **Deleted file** — remove all chunks where `file_path` matches.
- **Changed/added file** — delete existing chunks for that `file_path`,
  re-parse, re-chunk, re-enrich, re-embed, and re-insert.

Because chunk IDs are deterministic (file path + symbol name), this is a
straightforward delete-and-replace.

Rename detection is out of scope for MVP. Renames are treated as delete+add.

On success, the pipeline updates `commit_sha` in Postgres.

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

### 3.4 Vector store

**Qdrant** covers every embedding-adjacent need.

- **Vectors** — dense embeddings for semantic search.
- **Sparse vectors** — Qdrant generates BM25 sparse vectors natively for the lexical half of hybrid search.
- **Payload** — chunk metadata (`file_path`, `symbol_name`, `start_line`,
  `end_line`, `language`) stored per point, with indexed fields
  for fast filtering (e.g. language-scoped search).
- **Fusion** — dense + sparse results are combined server-side via the
  Query API's built-in RRF, no custom fusion code required (§2.2).

**Practical constraint:** Qdrant point IDs must be 64-bit unsigned integers
or UUIDs.

### 3.5 Conversation & repo-state store

**PostgreSQL** holds everything that isn't a vector: LangGraph conversation
checkpoints and repo metadata (`github_url`, `commit_sha`). Full design:
`docs/query_agent.md`.