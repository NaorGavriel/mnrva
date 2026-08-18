# Query Agent Component

Component 2 (`docs/architecture.md` §2.2). A LangGraph agent that answers
natural-language questions about one already-ingested repository.

Scope: single repo per Qdrant collection, multi-turn conversation, multiple tools.

## 2.1 Required tools

All four are bound to the agent node; the agent decides which to call.

- **`hybrid_search_tool(query: str, top_k: int = 10, language: str | None = None, kind: str | None = None)`**
  Wraps a new `chunks.search_chunks(client, collection_name, query_text, top_k, language=None, kind=None)`.
  Uses Qdrant's Query API: `prefetch=[Prefetch(query=embed_text(query_text), using=DENSE_VECTOR_NAME), Prefetch(query=Document(text=query_text, model="Qdrant/bm25"), using=SPARSE_VECTOR_NAME)]`,
  `query=FusionQuery(fusion=Fusion.RRF)`, optional `Filter` on `language`/
  `kind`.
  Returns payload + score per hit — `file_path`, `symbol_name`,
  `class_name`, `kind`, `start_byte`, `end_byte`, `raw_text`,
  `context_text`, `score`.

- **`whole_file_read_tool(file_path: str, start_line: int | None = None, end_line: int | None = None)`**
  Reads from the process's local clone (§2.3). Whole file or line slice.

- **`grep_search_tool(pattern: str, file_glob: str | None = None)`**
  `git grep -n --fixed-strings` inside the clone.
  One mechanism for both identifier lookups and error-string lookups.

- **`submit_answer(answer: str, evidence: list[Citation])`** — how the agent
  finishes a turn; there's no free-text final message. `Citation` =
  `{file_path, start_line, end_line}`. `evidence` is required and
  non-empty: the agent can't submit a claim with no cited source. One tool
  call per turn, same as the other three.

## 2.2 New `chunks.py` surface
- `search_chunks(...)` — §2.1, new.

## 2.3 Repo clone lifecycle (per process)

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
  §2.8 for why the alternatives were rejected.
  - **Cold start**: new `clone_full_repository(github_url, dest_dir) -> Path`
    in `repository_clone.py` — plain `git clone`, no `--depth`/`--filter`.
    The one intentionally expensive step, paid once per process at
    startup — off the conversation-serving path.
  - **Update**: new `update_repository(repo_path: Path, commit_sha: str) -> None`
    — `git fetch origin` (all branches by default, so no branch needs
    tracking) then `git reset --hard <commit_sha>`. Only files that
    actually changed get rewritten.
- The refresh pipeline (component 3) is what actually advances
  `commit_sha` in Postgres after each sync; every agent process
  independently converges to it the next time it checks.
- **New modules, `db_postgres.py` + `repo_metadata.py`** — same split as
  the Qdrant side: `db_postgres.py` owns the connection/pool (from
  `POSTGRES_URL`) and table setup; `repo_metadata.py` holds the CRUD for
  the repo-metadata row (`github_url`, `commit_sha`, `updated_at`) that
  uses it. `ingest_repository` writes the initial row; the refresh
  pipeline updates `commit_sha`; the agent only reads. Conversation state
  (§2.4) shares this same connection.
- **Concurrency gap**: `clone_repository` does `rmtree` then `clone` on the
  same path. If a re-clone is triggered while another conversation on the
  same process is mid-read against that path, reads can fail or tear. Not
  addressed here.

## 2.4 Conversation state

LangGraph's Postgres checkpointer persists `messages` per `thread_id`. `thread_id`
scopes *conversation* history only. `agent_messages.py` is a thin
wrapper handing back a configured `PostgresSaver` over `db_postgres.py`'s
connection — most of the actual logic is the library's.

```
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    answer_grade: Literal["good", "bad"] | None
    evaluation_reasoning: str | None
    retry_count: int
```

`evaluate_answer` (§2.5) reads the draft answer/evidence straight off the
most recent `submit_answer` tool call in `messages`, and the
hybrid-search-required gate scans `messages` the same way — neither is
duplicated into its own state field.

## 2.5 Graph nodes

```
START → sync_clone → agent → tools ─(hybrid_search / read_file / grep)──→ agent
                                │
                          (submit_answer)
                                ▼
                    [gate: hybrid_search called
                     since the last HumanMessage?]
                          │           │
                         no          yes
                          │           ▼
                          │     evaluate_answer ─(good)→ END
                          │           │
                          │   (bad, retry_count < MAX)
                          │           ▼
                          └──────→  agent    (loop; evaluator's reasoning
                                              injected as context)
                                    │
                          (bad, retry_count == MAX)
                                    ▼
                                   END   (answer stands, low-confidence note)
```

- **`sync_clone`** — not an LLM call; runs the check from §2.3.
- **`agent`** — LLM node (`AGENT_MODEL` env var, §2.7), bound to all four
  tools, one call per turn. Free to call `hybrid_search_tool`/
  `whole_file_read_tool`/`grep_search_tool` in any order and combination;
  finishes only by calling `submit_answer`. System prompt guides it on
  question intent (implementation / architecture / heuristics / a
  specific symbol / workflow) to help pick a tool — guidance only for
  now, not a structured decision.
- **`tools`** — `ToolNode` executing whatever the agent requested.
- **hybrid-search gate** — not an LLM call. Runs only after a
  `submit_answer` call; checks whether `hybrid_search_tool` was called
  since the most recent `HumanMessage`. If not, routes back to `agent`
  with an injected instruction to search before submitting — a
  code-enforced precondition, not a prompted one.
- **`evaluate_answer`** — LLM node, runs once the gate passes. Grades the
  submitted answer against the *user's original question* (not the
  retrieved chunks against the search query): `"good"` or `"bad"` plus
  reasoning on what the answer is missing when bad.

Once `retry_count` hits `MAX_RETRIES`, control still returns to `agent`
for one more pass, then ends — the graph never hard-fails on a bad
evaluation.

## 2.6 Life-cycle

1. Caller starts a conversation with some `thread_id`, against some
   already-running agent process.
2. `sync_clone` runs (§2.3) — cheap after the first turn.
3. `agent` reasons over the query, looping through `tools` and
   `evaluate_answer` until it submits an answer that passes grading (or
   the retry budget runs out).
4. Postgres checkpoints `messages`/`answer_grade`/`retry_count` under
   `thread_id` — visible to any process handling a follow-up on that
   `thread_id`, not just the one that handled turn 1.
5. Follow-up turn, possibly on a different process: prior `messages`
   restored from Postgres; that process's own `sync_clone` runs
   independently (§2.3).

No per-conversation cleanup step — the clone is process-scoped and
long-lived, not tied to any one conversation.

## 2.7 Constraints

- **Single repo per collection** — no `repo` filter on any tool.
- **`retry_count` cap** = 2.
- **LangGraph `recursion_limit`** — hard backstop on total node
  transitions per turn, beyond the retry cap.
- **`git grep`/`clone_repository` have no timeout** — pre-existing gap in
  `repository_clone.py`, inherited here.
- **Full-clone startup cost is repo-dependent and unbounded** — a repo
  with a large or bloat-heavy history (old binaries/assets still in the
  object database) makes a process's first clone slower and bigger, with
  no cap. Accepted for MVP.

## 2.8 Decisions & reasoning (recap)

- **`raw_text`/`context_text` added to the Qdrant payload** — makes a
  search hit self-contained; the agent isn't forced into a file read just
  to see the text it already matched on.
- **Per-process local clone + sha check, not a Postgres-backed file
  store** — considered and rejected storing file content in Postgres:
  it would have made `grep_search_tool` an unindexed regex scan over
  every file on every call, materially slower than `git grep` against an
  OS-cached local checkout. Postgres storage was solving a problem the sha check had already solved, while introducing a
  real performance regression on the tool that needs to be fast.
- **Repo metadata and conversation state in Postgres, not Qdrant** —
  Qdrant stays a pure vector/payload store; Postgres is the
  system-of-record for everything relational. Conversation checkpointing
  uses LangGraph's own Postgres checkpointer rather than a hand-rolled messages table.
- **Answer-level grading, not retrieval-level** — judges the final answer
  against the user's question, not the raw hits against the search query;
  relevant-looking hits don't guarantee the question got answered. No
  separate rewrite node — the retry loop reuses the ordinary `agent` node
  with the evaluator's reasoning added as context, since it already has
  full tool freedom.
- **`submit_answer` as a tool, not free text** — makes evidence a
  schema-enforced requirement instead of a prompted one, and reuses the
  same tool-call routing as the other three tools. Also settles the
  citation format: `file_path`/`start_line`/`end_line`.
- **Hybrid-search-before-submit is a code gate, not a prompt instruction**
  — guarantees the index gets consulted every turn, regardless of what
  the agent already appears to know from `grep`/file reads.
- **Full clone over shallow-refetch-by-SHA or blobless/partial clone** —
  the only option that's both host-agnostic (no server-side feature flag
  required, unlike SHA-in-want) and offline-safe after the initial sync
  (unlike blobless clone, which keeps depending on origin being reachable
  at every future checkout). Cost tradeoff — §2.7.

## 2.9 Known limitations / open decisions

- Per-process clone concurrency under simultaneous conversations on the
  same process — §2.3.
- Intent taxonomy (implementation / architecture / heuristics / symbol /
  workflow) is prompt guidance only for now — may become a structured
  decision later.
- Postgres schema for repo metadata (single row vs. keyed table) not
  spelled out — trivial either way at single-repo MVP scope.
- No mitigation designed for pathologically large repo histories (§2.7)
  — e.g. a `git backfill`-style incremental approach, if it ever matters.
