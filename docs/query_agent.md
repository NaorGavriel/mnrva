# Query Agent Component

Component 2 (`docs/architecture.md` §2.2). A LangGraph agent that answers
natural-language questions about one already-ingested repository.

Scope: single repo per Qdrant collection, multi-turn conversation, multiple tools.

## 2.1 Required tools

All three are bound to the agent node; the agent decides which to call.

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

## 2.2 New `chunks.py` surface
- `search_chunks(...)` — §2.1, new.

## 2.3 Repo clone lifecycle (per process)

There's no guarantee the query agent runs on the same machine as ingestion, or that there will be only one
query-agent process, so each agent maintains their own clone of the repository.

- Each process maintains one local clone, e.g.
  `repository_files/agent_clone/`, reused across every conversation that
  process handles.
- Each process sync it's existing (or non-existing) repository clone on start-up, clones the repository if required.
- At the start of each conversation, a `sync_clone()` step compares the
  clone's current commit (`repository_clone.get_current_commit_sha`)
  against `commit_sha` in Postgres. Mismatch (or no clone yet) →
  `clone_repository` re-clones. Match → no-op (cheap: one local `git
  rev-parse` plus one Postgres read).
- The refresh pipeline (component 3) is what actually advances
  `commit_sha` in Postgres after each sync; every agent process
  independently converges to it the next time it checks.
- **New modules, `db_postgres.py` + `repo_metadata.py`** — same split as
  the Qdrant side: `db_postgres.py` owns the connection/pool (from
  `POSTGRES_URL`) and table setup; `repo_metadata.py` holds the CRUD for
  the repo-metadata row (`github_url`, `commit_sha`, `updated_at`) that
  uses it. `ingest_repository` writes the initial row; the refresh
  pipeline updates `commit_sha`; the agent only reads. Conversation state
  (§2.4) shares `db_postgres.py`'s connection via its own `agent_messages.py`.
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
    retrieval_grade: Literal["relevant", "irrelevant"] | None
    rewrite_count: int
```

## 2.5 Graph nodes

```
START → sync_clone → agent ─(no tool calls)→ END
                        │
                 (tool calls present)
                        ▼
                      tools ──(last call wasn't hybrid_search)──→ agent
                        │
              (last call was hybrid_search)
                        ▼
                 grade_retrieval ─(relevant)→ agent
                        │
               (irrelevant, rewrite_count < MAX)
                        ▼
                  rewrite_query → tools (re-runs hybrid_search)
                        │
               (irrelevant, rewrite_count == MAX)
                        ▼
                       agent   (proceeds with a low-confidence note)
```

- **`sync_clone`** — not an LLM call. Runs the sha comparison (§2.3) once
  per conversation turn; usually a no-op after the first turn on a given
  process.
- **`agent`** — LLM node (`AGENT_MODEL` env var, §2.7), bound to all three
  tools. Calls a tool or emits the final answer.
- **`tools`** — `ToolNode` executing whatever the agent requested.
- **`grade_retrieval`** — LLM node, runs only after a `hybrid_search_tool`
  call. Judges whether the hits are relevant to the query that produced
  them.
- **`rewrite_query`** — LLM node, runs when grading fails and
  `rewrite_count < MAX_REWRITES` (proposed: 2). Rewrites/expands the query,
  re-invokes `hybrid_search_tool`. Once the budget is spent, control falls
  through to `agent`, which decides whether to try another tool or answer
  with a caveat - the graph never hard-fails on a bad retrieval.

## 2.6 Life-cycle

1. Caller starts a conversation with some `thread_id`, against some
   already-running agent process.
2. `sync_clone` checks/refreshes that process's clone (cheap after the
   first conversation of the process's lifetime, or after any
   refresh-pipeline run that moved `commit_sha`).
3. `agent` reasons over the query, looping through `tools`/
   `grade_retrieval`/`rewrite_query` until it emits an answer.
4. Postgres checkpoints `messages`/`retrieval_grade`/`rewrite_count` under
   `thread_id` — visible to any process handling a follow-up on that
   `thread_id`, not just the one that handled turn 1.
5. Follow-up turn, possibly on a different process: prior `messages`
   restored from Postgres; that process's own `sync_clone` runs
   independently (first-time clone if it's never handled this repo
   before, no-op if it has and is current).

No per-conversation cleanup step — the clone is process-scoped and
long-lived, not tied to any one conversation.

## 2.7 Constraints

- **Single repo per collection** — no `repo` filter on any tool.
- **`rewrite_count` cap** = 2.
- **LangGraph `recursion_limit`** — hard backstop on total node
  transitions per turn, beyond the rewrite cap.
- **`git grep`/`clone_repository` have no timeout** — pre-existing gap in
  `repository_clone.py`, inherited here.

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
- **Grading + query-rewrite node** — targets vague/paraphrased queries
  under-matching in hybrid search with a bounded, cheap LLM check, while
  leaving the agent free to try a different tool once the budget is spent.

## 2.9 Known limitations / open decisions

- Per-process clone concurrency under simultaneous conversations on the
  same process — §2.3.
- Exact `rewrite_count` value is proposals, not tuned.
- Citation format in the final answer (e.g. `file_path:start_line-end_line`)
  isn't specified yet — the agent's system prompt should require it, exact
  wording still open.
- Postgres schema for repo metadata (single row vs. keyed table) not
  spelled out — trivial either way at single-repo MVP scope.
