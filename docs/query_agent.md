# Query Agent Component

Component 2 (`docs/architecture.md` §2.2). A LangGraph agent that answers
natural-language questions about one already-ingested repository.

Scope: single repo per Qdrant collection, multi-turn conversation.
Corrective-RAG pipeline: fixed retrieve → grade → augment → generate →
evaluate stages, with one loop-back edge for re-retrieval.
**Status: design only, not yet implemented** — `nodes.py`, `tools.py`,
`agent_schemas.py` still implement an earlier ReAct-style agent/tools
loop and need rewriting to match.

## Graph

```mermaid
graph TD
    START([START]) --> EQ[evaluate_question]
    EQ --> RD[retrieve_documents]
    RD --> GD[grade_documents]
    GD -->|yes-labeled chunks| TL[tool: grep / read]
    TL --> GA[generate_answer]
    GA --> EA[evaluate_answer]
    EA -->|bad, attempts < 5<br/>reasoning appended to search_query| RD
    EA -->|good, or cap hit| END([END])
```

## Nodes

All LLM nodes run on `AGENT_MODEL` (env var).

1. **`evaluate_question`** (LLM, structured output) — reasons about the
   question: implementation, architecture, a heuristic/design decision, a
   specific named symbol, or a workflow. Produces `synthesized_query`
   (question + any added context — this, not the raw question, is what
   gets searched), `filters` (`language`/`kind`), and
   `expects_multiple_retrievals` — computed but currently unused by any
   edge (retrieval-attempt cap is a flat 5 regardless; revisit if that
   proves wasteful or insufficient).

2. **`retrieve_documents`** (not an LLM call) — calls `search_chunks`
   (§2.1) with `search_query` + `filters`. Hits merge into
   `retrieved_chunks`, keyed by chunk id, so a chunk returned by more
   than one attempt doesn't duplicate.

3. **`grade_documents`** (LLM, structured output, one call per
   newly-retrieved chunk) — labels each chunk not yet graded `yes`/`no`
   for relevance to the user's question. Labels persist across attempts;
   a chunk is graded once.

4. **`tool`** (LLM, bound to `grep_search_tool`/`whole_file_read_tool`,
   §2.1) — gets the question and the `yes`-labeled chunks as context,
   free to call either tool to fill in what the graded chunks don't
   cover (e.g. following an import, confirming a symbol definition).
   Runs once per pass through the graph — open question whether that's
   enough for cross-file chasing (`architecture.md` §2.2's "read file →
   notice import → read that file" mechanism), or whether this needs to
   be a small loop instead.

5. **`generate_answer`** (LLM, structured output) — question +
   `yes`-labeled chunks + tool output → `Answer` (§2.4).

6. **`evaluate_answer`** (LLM, structured output) — grades the generated
   `Answer` against the *original user question* (not the retrieved
   chunks against the search query — a distinct judgment from
   `grade_documents`'s per-chunk relevance). `good` → END. `bad`, under
   the attempt cap → loops back to `retrieve_documents` with the
   evaluator's reasoning appended to `search_query` (additive, doesn't
   replace `synthesized_query`/`filters`). `bad`, cap hit → END anyway,
   answer stands.

## Edges

1. `START → evaluate_question`
2. `evaluate_question → retrieve_documents`
3. `retrieve_documents → grade_documents`
4. `grade_documents → tool`
5. `tool → generate_answer`
6. `generate_answer → evaluate_answer`
7. `evaluate_answer → retrieve_documents` — `bad` and `retrieval_attempts < 5`
8. `evaluate_answer → END` — `good`, or `retrieval_attempts == 5`

## 2.1 Retrieval & tools

- **`search_chunks`** (`chunks.py`) — called directly by
  `retrieve_documents`, not agent-invoked. Uses Qdrant's Query API:
  `prefetch=[Prefetch(query=embed_text(query_text), using=DENSE_VECTOR_NAME), Prefetch(query=Document(text=query_text, model="Qdrant/bm25"), using=SPARSE_VECTOR_NAME)]`,
  `query=FusionQuery(fusion=Fusion.RRF)`, optional `Filter` on
  `language`/`kind`. Returns payload + score per hit — `file_path`,
  `symbol_name`, `class_name`, `kind`, `start_byte`, `end_byte`,
  `raw_text`, `context_text`, `score`.

- **`whole_file_read_tool(file_path, start_line=None, end_line=None)`**
  — reads from the process's local clone (§2.2). Whole file or line
  slice. Bound to the `tool` node.

- **`grep_search_tool(pattern, file_glob=None) -> list[GrepMatch]`** —
  `git grep -n --fixed-strings` inside the clone, parsed into
  `GrepMatch = {file_path, line_number, line_text}` per match. lets
  `generate_answer` build a `Citation` directly off a match without
  parsing. Bound to the `tool` node.

## 2.2 Repo clone lifecycle (per process)

There's no guarantee the query agent runs on the same machine as ingestion, or that there will be only one
query-agent process, so each agent maintains their own clone of the repository.

- Each process maintains one local clone, e.g.
  `repository_files/agent_clone/`, reused across every conversation that
  process handles.
- `sync_clone()` runs at process startup, comparing the clone's current
  commit (`repository_clone.get_current_commit_sha`) against `commit_sha`
  in Postgres: no clone yet → **cold start**, mismatch → **update**,
  match → no-op (cheap: one local `git rev-parse` + one Postgres read).
- **Clone strategy: full clone, not shallow, not partial/blobless** — see
  §2.6 for why the alternatives were rejected.
  - **Cold start**: new `clone_full_repository(github_url, dest_dir) -> Path`
    in `repository_clone.py` - plain `git clone`. The one expensive step, paid once per process at startup.
  - **Update**: `update_repository(repo_path: Path, commit_sha: str) -> None`
    — `git fetch origin` then `git reset --hard <commit_sha>`. Only files that
    actually changed get rewritten.
- **New modules, `db_postgres.py` + `repo_metadata.py`**: 
  `db_postgres.py` owns the connection/pool and table setup.
  `repo_metadata.py` holds the CRUD for the repo-metadata row (`github_url`, `commit_sha`, `updated_at`). `ingest_repository` writes the initial row; the refresh
  pipeline updates `commit_sha`; the agent only reads.

## 2.3 Conversation & agent state

LangGraph's Postgres checkpointer persists all of `AgentState` per
`thread_id`. `thread_id` scopes *conversation* history only.
`agent_messages.py` is a thin wrapper handing back a configured
`PostgresSaver` over `db_postgres.py`'s connection.

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    search_query: str
    search_filters: dict  # {"language": str | None, "kind": str | None}
    expects_multiple_retrievals: bool
    retrieved_chunks: Annotated[dict[str, ChunkSearchResult], merge_by_key]  # keyed by chunk id
    chunk_relevance: dict[str, Literal["yes", "no"]]  # keyed by chunk id
    retrieval_attempts: int
    answer: Answer | None
    answer_grade: Literal["good", "bad"] | None
    evaluation_reasoning: str | None
```

`retrieved_chunks` needs a custom merge reducer (parallel to
`add_messages` on `messages`) — LangGraph's default TypedDict field
semantics overwrite on each node return, which would drop everything
from prior attempts.

`Answer = {text: str, citations: list[Citation]}`. `Citation =
{file_path, start_line, end_line, citation_text}` — `citation_text` is
the quoted source excerpt itself (code or prose), not just its location.

## 2.4 Life-cycle

1. Caller starts a conversation with some `thread_id`, against some
   already-running agent process.
2. `sync_clone` runs.
3. The graph runs `evaluate_question → retrieve_documents →
   grade_documents → tool → generate_answer → evaluate_answer` once,
   correcting via re-retrieval (up to 5 attempts) if the answer grades
   bad.
4. Postgres checkpoints `AgentState` under `thread_id` — visible to any
   process handling a follow-up on that `thread_id`, not just the one
   that handled turn 1.
5. Follow-up turn, possibly on a different process: prior state restored
   from Postgres; that process's own `sync_clone` runs independently.


## 2.5 Constraints

- **`retrieval_attempts` cap** = 5.

## 2.6 Decisions & reasoning (recap)

- **`raw_text`/`context_text` added to the Qdrant payload** — makes a
  search hit self-contained; the agent isn't forced into a file read just
  to see the text it already matched on.
- **Per-process local clone + sha check, not a Postgres-backed file
  store** — considered and rejected storing file content in Postgres:
  it would have made `grep_search_tool` an unindexed regex scan over
  every file on every call, materially slower than `git grep` against an
  OS-cached local checkout.
- **Repo metadata and conversation state in Postgres, not Qdrant** —
  Qdrant stays a pure vector/payload store; Postgres is the
  system-of-record for everything relational. Conversation checkpointing
  uses LangGraph's own Postgres checkpointer rather than a hand-rolled messages table.
- **Per-chunk grading before generation, answer-level grading after** —
  `grade_documents` filters what `generate_answer` sees; `evaluate_answer`
  separately judges whether the resulting answer actually holds up
  against the user's question. Relevant-looking hits don't guarantee the
  question got answered.
- **Corrective re-retrieval on `evaluate_answer`, not `grade_documents`**
  — a document-relevance check alone can pass (some chunks are on-topic)
  while the synthesized answer still falls short; gating the loop on the
  answer catches that, and folding the evaluator's reasoning into the
  next `search_query` gives the retry a concrete reason to look
  different from the first attempt.
- **Full clone over shallow-refetch-by-SHA or blobless/partial clone** —
  the only option that's both host-agnostic and offline-safe after the initial sync
  (unlike blobless clone, which keeps depending on origin being reachable at every future checkout).

## 2.7 Known limitations / open decisions

- `tool` node runs once per pass — unresolved whether that's sufficient
  for multi-hop cross-file questions (§ Nodes, item 4).
- `expects_multiple_retrievals` computed but unused by any edge (§ Nodes,
  item 1).
