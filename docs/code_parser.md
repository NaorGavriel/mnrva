# Code Parsing Component

## 1.1 `models.py`
Shared data shapes used by both the parsing stage and the downstream (enrichment/embedding) stages.

* `Chunk` — dataclass:
  - `id: str` — **stable identity, not a content fingerprint.** `uuid5(CHUNK_NAMESPACE, file_path + kind + class_name + symbol_name)`. Deterministic and content-independent, so editing a chunk's body never changes its id — the Refresh & Sync Pipeline can upsert directly by id with no search-then-delete step, and never leaves orphaned points behind in Qdrant. `uuid5` specifically (not a raw hex digest) because Qdrant point ids are constrained to 64-bit unsigned integers or UUIDs only — a plain hash string is rejected.
    - `CHUNK_NAMESPACE` is a hardcoded constant UUID (generate once with `uuid4()`, never regenerate) — changing it shifts every id in the system.
    - Id-construction inputs by chunk shape:

      | chunk type | class_name | symbol_name |
      |---|---|---|
      | top-level function | `""` | function name |
      | method | class name | method name |
      | class (container) | class name | `""` |
  - `content_hash: str` — `sha256(raw_text)` hex digest. A fingerprint, not an identifier — lives in the Qdrant payload, compared on refresh to decide whether re-embedding is needed. Unlike `id`, this is expected to change whenever the chunk's body changes.
  - `path: Path`
  - `language: str`
  - `kind: Literal["class", "function"]`
  - `class_name: str` — `""` for top-level functions and other non-nested chunks; the class name for methods and for the class chunk itself. Also feeds `id` construction above.
  - `symbol_name: str` — the function/method/class name, used in `id` construction and independently useful for display and enrichment.
  - `raw_text: str` — decoded UTF-8. Non-UTF-8 files are filtered upstream in `repository_parser.py`.
  - `start_byte: int`, `end_byte: int`
  - `parent_id: str | None` — id of the enclosing container chunk (e.g. the class a method lives in). `None` for top-level functions, nested helper functions with no enclosing container, and for container chunks themselves.
  - `context_text: str | None = None` — populated later by `enrichment.py`, not by `code_parser.py`.
  - `embedding: list[float] | None = None` — populated later by `embedding.py`, not by `code_parser.py`.

## 1.2 `languages.py`
Single source of truth for language/extension configuration — avoids the extension list existing separately in the ingestion allowlist and the tree-sitter routing table.

* `LANGUAGE_CONFIG` — `extension -> LanguageConfig`, per §3.3 of the main architecture doc:
  - `language: str` — canonical name (`"python"`, `"typescript"`, `"csharp"`, ...). Joins into the language names `registry.py`'s `LanguageRegistry` builds a `Parser` for — see §1.3.
  - `container_node_types: set[str]` — node types chunked as a whole unit (e.g. `{"class_definition"}` for Python)
  - `unit_node_types: set[str]` — node types chunked individually, matched at any depth in the tree, including inside containers and inside other units. Nested helper functions are chunked too — this is intentional.
* `get_language(path: Path) -> str | None`
* `is_code_file(path: Path) -> bool`
* `is_prose_file(path: Path) -> bool`

**Known gap to resolve before implementing JS/TS support:** a named arrow function (`const handleClick = () => {...}`) is not its own node type — it's an `arrow_function` nested inside a `variable_declarator`. A flat `unit_node_types` set will not catch this pattern, which is common in real JS/TS code. Decide before shipping JS/TS: accept under-chunked arrow functions in v1, or extend `unit_node_types` to express a declarator-shape match rather than a bare node type.

## 1.3 `registry.py`
Loads and caches tree-sitter `Language`/`Parser` objects. This is the only module that imports grammar packages directly — everything downstream depends on the `GrammarRegistry` interface, never on the concrete packages.

* `GrammarRegistry` (Protocol) — `get_parser(language: str) -> Parser`
* `LanguageRegistry` (implementation) — builds every supported language's `Parser` eagerly in `__init__`, from static top-level imports:
  ```python
  import tree_sitter_python as tspython
  import tree_sitter_javascript as tsjavascript
  import tree_sitter_typescript as tstypescript
  import tree_sitter_c_sharp as tscsharp
  from tree_sitter import Language, Parser

  class LanguageRegistry:
      def __init__(self):
          self._parsers: dict[str, Parser] = {
              "python": Parser(Language(tspython.language())),
              "javascript": Parser(Language(tsjavascript.language())),
              "typescript": Parser(Language(tstypescript.language_typescript())),
              "tsx": Parser(Language(tstypescript.language_tsx())),
              "csharp": Parser(Language(tscsharp.language())),
          }

      def get_parser(self, language: str) -> Parser:
          return self._parsers[language]
  ```
  All values are the same generic `tree_sitter.Parser` class — each instance just has a different grammar bound to it. No per-language subclasses; language-specific behavior stays entirely in `LANGUAGE_CONFIG`'s node types (§1.2), read by `code_parser.py`, never by `registry.py`.
  Dict keys must match the `language` values used in `LANGUAGE_CONFIG` (§1.2) — a real cross-module invariant, caught by the §1.7 contract test.
  Constructed once per pipeline run — any broken or missing grammar fails immediately at that point, not on first use — and passed down explicitly (dependency injection), never a module-level singleton, so tests can substitute a fake registry.

## 1.4 `repository_clone.py`

* `clone_repository(github_url: str, dest_dir: Path) -> Path` — shell out via `subprocess.run([...])` with an argument list, never a shell string, since `github_url` is user-provided input. For ingestion, always a fresh shallow clone (`--depth=1`) — there's no prior state to diff against yet, so there's nothing history would buy here.
* `get_current_commit_sha(repo_path: Path) -> str` — cheap to capture at ingestion time; store it alongside the repo's chunks will be used in ReSync component.

Clone depth for *re*-syncing an already-ingested repo is a Refresh & Sync concern, not an ingestion one — see §1.8.

## 1.5 `repository_parser.py`
Returns the filtered file list for ingestion.

* `list_source_files(repo_path: Path) -> list[Path]` — `git ls-files`, filtered through the extension allowlist (`languages.py`), a filename denylist (e.g. `package-lock.json`), and a file-size cutoff.
* `_passes_size_cutoff(path: Path, max_bytes: int) -> bool`
* `_is_denied_filename(path: Path) -> bool`

## 1.6 `code_parser.py`
Parsing only — no LLM calls, no embedding calls. Split into a thin file-level orchestrator and a pure, language-agnostic extraction function.

* `parse_code_file(path: Path, language: str, registry: GrammarRegistry) -> list[Chunk]`
  - Reads the file, gets a `Parser` from `registry`, parses it to a `Tree`, delegates to `extract_chunks` with that language's `LANGUAGE_CONFIG` entry.
  - `registry` is a parameter, not a global — this is the seam that lets tests inject a fake registry instead of loading real grammars.

* `extract_chunks(tree: Tree, source: bytes, config: LanguageConfig, path: Path) -> list[Chunk]`
  - **Pure function** — no disk I/O, no grammar loading. Same `(tree, source, config, path)` always produces the same chunks. This is what makes it unit-testable against inline source strings, with no fixture files required.
  - Two passes, deliberately overlapping:
    1. Query `container_node_types` → one chunk per container (e.g. each class), `kind="class"`.
    2. Query `unit_node_types` at any depth → one chunk per function/method, `kind="function"`, with `parent_id` set to the nearest enclosing container chunk's id, or `None` if there isn't one.
  - A method produces both its own chunk *and* contributes to its parent class's chunk text — intentional duplication, so retrieval works at both class and method granularity.
  - Files with no classes (only top-level functions) work identically — `parent_id` is simply `None` throughout. No special-casing needed for "files that only have methods in them."

* `_query_nodes(language: Language, root: Node, node_types: set[str]) -> list[Node]`
  - Matches via tree-sitter's native `Query`/`QueryCursor` engine : builds a query string of one bare `(node_type) @match` pattern per entry in `node_types`, compiles it once per `(language, node_types)` pair (cached in `_get_query`/`_QUERY_CACHE`), then runs it with `QueryCursor(query).captures(root)`. Measured ~40% faster per call than a manual recursive walk on this repo's own `code_parser.py`; the C-side match loop wins over Python-level `node.children` recursion. A manual-walk version was implemented and benchmarked first and is intentionally not kept — see git history if the comparison needs revisiting.
* `_nearest_ancestor(node: Node, container_node_types: set[str]) -> Node | None`
* `_make_chunk(node: Node, source: bytes, path: Path, language: str, kind: str, class_name: str, parent_id: str | None) -> Chunk`
* `_extract_imports(tree: Tree, source: bytes, config: LanguageConfig) -> list[str]` — driven by an `import_node_types` entry in `LANGUAGE_CONFIG`, not by per-language branches inside this function.

**Explicitly out of scope for this module** — separate modules, separate failure domains:
* `enrichment.py` — `enrich_chunks(chunks: list[Chunk]) -> list[Chunk]`, populates `context_text` via LLM calls.
* `embedding.py` — `embed_chunks(chunks: list[Chunk]) -> list[Chunk]`, populates `embedding` via batched embedding calls.

Per-file orchestration composes all three: `parse_code_file` → `enrich_chunks` → `embed_chunks` → store.

## 1.7 Testing strategy

* **`extract_chunks`** (core of the test suite) — parametrized across every configured language using small inline source snippets, no fixture files needed. Include one invariant test that holds for every language: `chunk.raw_text == source[chunk.start_byte:chunk.end_byte]`.
* **`LANGUAGE_CONFIG` contract test** — for every configured extension, parse a small canary snippet containing at least one class and one function, and assert a sane number of chunks come out. Catches config typos (wrong node type name, a `LANGUAGE_CONFIG` language with no matching parser entry in `registry.py`) at commit time instead of in production ingestion.
* **`parse_code_file`** — integration-style tests using `tmp_path`: a real file on disk, a non-UTF-8 file, an empty file, an unrecognized extension. Kept small since most logic is already covered via `extract_chunks`.
* **`registry.py`** — test that repeated `get_parser` calls for the same language return the cached instance (identity check); test the injected-fake-registry path used by `parse_code_file`'s error-handling tests.
* **`repository_parser.py`** — `_passes_size_cutoff` / `_is_denied_filename` get plain unit tests, no filesystem involved. `list_source_files` gets an integration test against a real `git init`'d fixture repo in `tmp_path`, not a mocked `subprocess` — git's actual output format is what needs verifying.

## 1.8 Known limitations / open decisions to resolve before implementation

* **JS/TS arrow functions** — see §1.2. Needs a decision before JS/TS support ships.
* **Parse-error policy** — what happens when `tree.root_node.has_error` is true (real syntax errors, or a file that only superficially matches its extension)? Skip the file, emit partial chunks, or flag it for review?
* **`raw_text` encoding** — confirmed as decoded `str`, with an explicit fallback policy needed for files that fail UTF-8 decoding.
* **Oversized chunks** — a very large function or class can exceed typical embedding context windows. Not urgent, but should be a tracked follow-up rather than a silent truncation discovered in production.
* **Nested closures** — resolved: nested helper functions are chunked individually regardless of depth, no "nearest chunkable ancestor" restriction.