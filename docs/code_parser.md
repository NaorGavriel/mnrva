# Code Parsing Component

## 1.1 `models.py`

* `Chunk` — `id`, `content_hash`, `path: PurePath`, `language`, `kind`, `class_name`, `symbol_name`, `raw_text`, `start_byte`, `end_byte`, `parent_id`, `context_text=None`, `embedding=None`.
  - `path` is `PurePath`, `code_parser.py` (§1.5) stores it as a `PurePosixPath`, so `str(chunk.path)` is always forward-slash regardless of the ingesting machine's OS.
* `make_chunk_id(path: PurePath, kind, class_name, symbol_name)` → `uuid5(CHUNK_NAMESPACE, ...)`. Deterministic, content-independent — refresh can upsert by id, no search-then-delete. `class_name`/`symbol_name` are `""` for top-level functions / class chunks respectively. 
* `make_content_hash(raw_text)` → sha256 hex digest. Compared on refresh to decide re-embedding.
* `chunk_retrieval_text(chunk)` → `context_text + raw_text` if enriched, else `raw_text`. Shared by embedding and BM25 so both see identical content.
* `ParsedFile` — `chunks`, `source`, `imports`.

## 1.2 `languages.py`

* `LANGUAGE_CONFIG: dict[ext, LanguageConfig]` — `language`, `container_node_types`, `unit_node_types`, `import_node_types`. Single source of truth for extension routing.
* `get_language`, `is_code_file`, `is_prose_file`.
* **Open**: JS/TS named arrow functions (`const f = () => {}`) aren't their own node type — nested in `variable_declarator`. A flat `unit_node_types` won't catch them; decide before shipping JS/TS.

## 1.3 `registry.py`

* `GrammarRegistry` (Protocol) — `get_parser(language) -> Parser`.
* `LanguageRegistry` — builds every `Parser` eagerly in `__init__`. Dict keys must match `LANGUAGE_CONFIG`'s `language` values.
* Always passed as a DI parameter, never a global — lets tests inject a fake.

## 1.4 `repository_clone.py` / `repository_parser.py`

Full design: `docs/repository_clone.md`.

## 1.5 `code_parser.py`

Parsing only — no LLM/embedding calls.

* `parse_code_file(read_from: Path, language: str, registry: GrammarRegistry, *, repo_root: Path | None = None) -> ParsedFile` — reads, parses, delegates to `extract_chunks`. Identity baked into `chunk.id` is `read_from.relative_to(repo_root)` if given, else `read_from`, converted to a `PurePosixPath` — keeps ids stable across re-clones regardless of where the scratch clone lives on disk, and regardless of the ingesting machine's OS.
* `extract_chunks(tree, source, config, path: PurePath) -> list[Chunk]` — pure. Two passes: containers first, then units at any depth (nested helpers included), `parent_id` set to the nearest enclosing container.
* `_query_nodes` — tree-sitter `Query`/`QueryCursor`, compiled query cached per `(language, node_types)`.
* `_nearest_ancestor`, `_make_chunk`, `_extract_imports`.
* Out of scope here: `enrichment.enrich_chunks(chunks, source, imports)` (LLM calls), `embeddings.embed_chunks(chunks)` (embeds `chunk_retrieval_text`, one `embed_text` call per chunk). Composed by `repository_ingester.py` — §1.8.

## 1.6 Testing strategy

* `extract_chunks` — parametrized per language, inline snippets. Invariant: `raw_text == source[start_byte:end_byte]`.
* `LANGUAGE_CONFIG` contract test — canary snippet per extension, catches config/registry typos at commit time.
* `parse_code_file` — `tmp_path` integration tests: real file, non-UTF-8, empty, unrecognized extension.
* `registry.py` — cached-instance identity check; fake-registry injection path.

## 1.7 Known limitations / open decisions

* JS/TS arrow functions — §1.2.
* Parse-error policy — what happens when `tree.root_node.has_error`? Undecided.
* `raw_text` encoding — non-UTF-8 fallback undecided.
* Oversized chunks — no size/truncation policy yet.
* Nested closures — resolved: chunked individually regardless of depth.
* **Enrichment/embedding throughput** — `enrich_chunks`/`embed_chunks` each make one sequential, unbatched API call per chunk (confirmed: ~127 chunks took several minutes end to end, dominated by `enrich_chunks`'s chat-completion calls). Not fixed yet; candidate optimizations, roughly in order of expected impact:
  - Parallelize both via a thread pool (`concurrent.futures.ThreadPoolExecutor`) — I/O-bound calls, no change to what's sent per call.
  - Batch `embed_chunks` — OpenAI's embeddings endpoint accepts a list of inputs in one request; currently one `embed_text` call per chunk.

## 1.8 `repository_ingester.py`

Component 1 entry point.

* `ingest_repository(github_url: str, registry: GrammarRegistry | None = None, client: QdrantClient | None = None) -> str` — `ensure_collection` → `clone_repository` → prune → for each source file that's code: `parse_code_file` (`repo_root=repo_path`) → `enrich_chunks` → `embed_chunks` → `upsert_chunks` → `delete_repository`. Returns the ingested commit sha. `registry`/`client` injectable for tests.
* **Gaps**: prose files skipped (`prose_parser.py` doesn't exist); no parse-error handling (one bad file aborts the run — §1.7); commit sha isn't durably stored anywhere a future resync process could read it — see `docs/repository_clone.md`.
