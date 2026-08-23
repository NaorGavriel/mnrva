# Query Agent Component

Component 2 (`docs/architecture.md` §2.2). A LangGraph agent that answers
natural-language questions about one already-ingested repository.

Scope: single repo per Qdrant collection, multi-turn conversation.
Corrective-RAG pipeline: fixed retrieve → grade → generate → evaluate
stages, with one loop-back edge for re-retrieval.

## Graph

```mermaid
graph TD
    START([START]) --> BCW[build_conversation_window]
    BCW --> EQ[evaluate_question]
    EQ --> RD[retrieve_documents]
    RD --> GD[grade_documents]
    GD -->|yes-labeled chunks| GA[generate_answer]
    GA --> EA[evaluate_answer]
    EA -->|bad, attempts < cap<br/>reasoning appended to search_query| RD
    EA -->|good, or cap hit| PM[persist_agent_message]
    PM --> END([END])
```

## Nodes

All LLM nodes run on `AGENT_MODEL` (env var).

1. **`build_conversation_window`** (not an LLM call) — turns the
   just-completed previous turn into one `conversation_window` entry
   (question, answer, and `cited_context` — the previous answer's cited
   chunks re-fetched live via `get_chunks_by_id`, §2.1, so a follow-up
   grounds in what the code says *now*), appends it, and trims to the
   most recent `CONVERSATION_WINDOW_TURNS` (env var). No-op on the first
   turn. Also formats the window into `conversation_history`, a single
   text block every other LLM node in this turn prepends to its own
   prompt, computed once here.

2. **`evaluate_question`** (LLM, structured output) — reasons about the
   question: implementation, architecture, a heuristic/design decision, a
   specific named symbol, or a workflow. Produces `synthesized_query`
   (question + any added context - this is what
   gets searched), `filters` (`language`).

3. **`retrieve_documents`** (not an LLM call) — calls `search_chunks`
   (§2.1) with `search_query` + `filters`. Hits merge into
   `retrieved_chunks`, keyed by chunk id, so a chunk returned by more
   than one attempt doesn't duplicate.

4. **`grade_documents`** (LLM, structured output, one call per
   newly-retrieved chunk) — labels each chunk not yet graded `yes`/`no`
   for relevance to the user's question. Labels persist across attempts;
   a chunk is graded once.

5. **`generate_answer`** (LLM, structured output) — question +
   `yes`-labeled chunks → `Answer` (§2.2). Each chunk shown to the model
   includes its real `chunk_id`, which the model must copy into its
   `Citation` (not invent) — `build_conversation_window` depends on that
   id being real to re-fetch the chunk later.

6. **`evaluate_answer`** (LLM, structured output) — grades the generated
   `Answer` against the *original user question* (not the retrieved
   chunks against the search query — a distinct judgment from
   `grade_documents`'s per-chunk relevance). `good` or cap hit →
   `persist_agent_message`. `bad`, under the attempt cap → loops back to
   `retrieve_documents` with the evaluator's reasoning appended to
   `search_query` (additive, doesn't replace `synthesized_query`/`filters`).

7. **`persist_agent_message`** (not an LLM call) — runs once per turn,
   after `evaluate_answer` reaches a terminal state. Appends an
   `AIMessage` built from `state.answer` to `messages` (§2.2) — the only
   node that writes to `messages` on the answer side of a turn; the
   `HumanMessage` for the question is appended by the caller before
   `graph.invoke()` (§2.3).

## Edges

1. `START → build_conversation_window`
2. `build_conversation_window → evaluate_question`
3. `evaluate_question → retrieve_documents`
4. `retrieve_documents → grade_documents`
5. `grade_documents → generate_answer`
6. `generate_answer → evaluate_answer`
7. `evaluate_answer → retrieve_documents` — `bad` and `retrieval_attempts < cap`
8. `evaluate_answer → persist_agent_message` — `good`, or cap hit
9. `persist_agent_message → END`

## 2.1 Retrieval

- **`search_chunks`** (`chunks.py`) — called directly by
  `retrieve_documents`. Uses Qdrant's Query API.

- **`get_chunks_by_id`** (`chunks.py`) — a lookup, not a search: one
  batched Qdrant `retrieve()` by chunk id. Called by
  `build_conversation_window`.

## 2.2 Conversation & agent state

LangGraph's Postgres checkpointer persists all of `AgentState` per
`thread_id`. `thread_id` scopes *conversation* history only.
`agent_messages.py` is a thin wrapper handing back a configured
`PostgresSaver` over `db_postgres.py`'s connection.

```python
class TurnContext(TypedDict):
    question: str
    answer: str
    cited_context: str  # formatted from cited chunks' context_text, "" if none cited

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    conversation_window: list[TurnContext]
    conversation_history: str
    search_query: str
    search_filters: dict  # {"language": str}
    expects_multiple_retrievals: bool
    retrieved_chunks: dict[str, ChunkSearchResult]
    chunk_relevance: dict[str, Literal["yes", "no"]]
    retrieval_attempts: int
    answer: Answer | None
    answer_grade: Literal["good", "bad"] | None
    evaluation_reasoning: str | None
```

Only `messages` needs the `add_messages` reducer — it's written across
turns by two different writers (the caller, `persist_agent_message`).
Every other field, including `conversation_window`, has exactly one
writer node, which manages its own merge/trim in plain code; LangGraph's
default overwrite-on-return semantics are already correct for those, so
no custom reducer is needed.

`Answer = {text: str, citations: list[Citation]}`. `Citation =
{chunk_id, file_path, start_line, end_line, citation_text}` —
`citation_text` is the quoted source excerpt itself (code or prose), not
just its location. Each turn's answer flattens to:

```python
AIMessage(
    content=answer.text,
    additional_kwargs={"citations": [c.chunk_id for c in answer.citations]},
)
```

## 2.3 Life-cycle

1. Caller starts a conversation with some `thread_id`, against some
   already-running agent process, appending a `HumanMessage(question)` to
   `messages` before calling `graph.invoke()`.
2. The graph runs `build_conversation_window → evaluate_question →
   retrieve_documents → grade_documents → generate_answer →
   evaluate_answer` once, correcting via re-retrieval (up to the attempt
   cap) if the answer grades bad, then `persist_agent_message`.
3. Postgres checkpoints `AgentState` under `thread_id` — visible to any
   process handling a follow-up on that `thread_id`, not just the one
   that handled turn 1.
4. Follow-up turn, possibly on a different process: prior state restored
   from Postgres.

## 2.4 Constraints

- **`retrieval_attempts` cap** — flat cap enforced in `graph.py`
  (`RETRIEVAL_ATTEMPTS_CAP`).
- **`CONVERSATION_WINDOW_TURNS` cap** (env var) — bounds
  `conversation_window`; oldest turn evicted once over the cap.

## 2.5 Decisions & reasoning (recap)

- **`raw_text`/`context_text` added to the Qdrant payload** — makes a
  search hit self-contained.
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