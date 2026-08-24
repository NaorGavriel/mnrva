# Plan: Rate-Limit-Aware, Repo-Wide Enrichment

## Goal

Make ingestion throughput scale with repo size (targeting ~100x current
chunk volumes) without hardcoding OpenAI rate limits, by replacing the
per-file sequential enrichment loop with a repo-wide, concurrency-bounded
pipeline that adapts to whatever tier the account is on.

## Current state

- `ingest_repository` (`repository_ingester.py:25`) is a single per-file
  loop: parse → `enrich_chunks` (1 sequential LLM call per chunk) → `embed_chunks`
  (already batched, `EMBEDDING_BATCH_SIZE` per call) → `upsert_chunks`.
- `enrich_chunks` (`enrichment.py:29`) calls `_generate_context` once per
  chunk, synchronously, in a `for` loop. Each call's prompt embeds the
  **entire file source** plus imports, not just the chunk (`enrichment.py:13-26`).
- No rate-limit config exists today.
- `docs/architecture.md` §2.1 documents the pipeline as strictly
  per-file, this plan changes that shape and the doc needs to change with it.

## Why whole-file-context matters here

Because each enrichment call re-sends the full file, token cost per file
scales with `file_tokens × chunk_count`, not `chunk_count` alone. This is
the main argument for batching a file's chunks into one call (see below):
it eliminates the duplication rather than just parallelizing it.

## Enrichment shape = batch per file

One LLM call enriches **all of a file's chunks at once**: prompt includes
the file source/imports once, plus the file's chunks, and asks for
structured output covering all of them. File source is sent once
regardless of chunk count — this is what fixes the duplication problem,
not just spreads it across threads.

### Request/response schema

Chunks are addressed by **position in the call, not chunk_id**. Each
chunk in a call's prompt gets a local `index` (0-based, matching its
position in that call's chunk list); the response echoes `index`, not the
UUID. Rationale: a chunk_id (`uuid5`) is ~9-12 tokens to include in the
prompt and again in every response entry, and is something the model
could transcribe wrong; a small integer is ~1 token and structurally
can't be mistyped into an unrelated valid-looking id. The caller maps
`index → chunk` locally from the request list it already built — no
string matching needed.

Prompt shape (per call):

```
<file path="{path}">
imports:
{imports_block}

{source}
</file>

<chunks>
[0] <chunk symbol="{symbol_name}">
{raw_text}
</chunk>
[1] <chunk symbol="{symbol_name}">
{raw_text}
</chunk>
...
</chunks>

For each chunk above, give a short (1-2 sentence) context situating it
within the file, for the purpose of improving search retrieval of the
chunk. Return one entry per chunk index.
```

Response, via OpenAI Structured Outputs (`response_format` json_schema,
`strict: true`):

```json
{
  "type": "object",
  "properties": {
    "contexts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "index": {"type": "integer"},
          "context": {"type": "string"}
        },
        "required": ["index", "context"],
        "additionalProperties": false
      }
    }
  },
  "required": ["contexts"],
  "additionalProperties": false
}
```

A dynamic `{chunk_id: context}` map.

Rejected alternative: per-chunk calls grouped by file (parallelize today's
shape without changing the prompt). Smaller diff, but doesn't address
token duplication — at 100x scale that duplication is the dominant cost,
so it doesn't serve the actual goal (scaling under a fixed/shifting rate
limit budget) as well as batching does.

### Per-call chunk cap (large files split into multiple calls)

A single call can't take a file's chunks unbounded — a file with many
chunks risks pushing the response near output-token limits. New config
`ENRICHMENT_MAX_CHUNKS_PER_CALL`. When a worker takes on a file:

- If `len(file.chunks) <= ENRICHMENT_MAX_CHUNKS_PER_CALL`: one call, as above.
- Otherwise: split the file's chunks into `ceil(len(chunks) /
  ENRICHMENT_MAX_CHUNKS_PER_CALL)` sub-batches, issued as separate calls.
- Sub-batch calls for the same file are issued **sequentially by the same
  worker**, not fanned out further — this keeps them adjacent in time,
  preserving OpenAI's prefix-cache hit rate on the shared file-source
  prefix across the file's sub-batches (same reasoning as the earlier
  cache-locality concern, just scoped to within-file now that files
  themselves may need >1 call).
- Chunk-count is the split trigger here (matches how the limit was
  specified); a token-aware split (accounting for large individual
  chunks, not just count) is a possible later refinement, not required
  for v1.

### Validation

Each call's response `index` set is diffed against the expected
`{0, ..., n-1}` for that call:

- Missing index → that chunk's enrichment failed; log it (file, chunk_id
  resolved locally from the index, call number if the file was split) and
  mark it for retry.
- Out-of-range/duplicate index → log and drop; likely a model error, not
  something to silently trust.
- This makes partial failure detectable and scoped to the specific
  sub-batch, not "something in this large file is wrong, figure out what."

Retry policy: retry only the missing chunks, as a new, smaller call
containing just those chunks (freshly re-indexed `0..k-1` for that retry
call, mapped back to the original chunks via the caller's own list — not
the original call's indices). A partial failure on a 40-chunk file costs
one small follow-up call, not 39 chunks' worth of redundant spend.

## Shared building block: rate limiter

New module `rate_limiter.py` (one concern: pacing outbound OpenAI calls —
per CLAUDE.md's "one concern per module").

- Async token-bucket, two buckets: requests/minute and tokens/minute.
- Sized from env: `OPENAI_MAX_REQUESTS_PER_MINUTE`, `OPENAI_MAX_TOKENS_PER_MINUTE`.
  Moving tiers = editing `.env`, not code.
- Callers `await` capacity (both buckets) before issuing a call, passing an
  estimated token cost for that call — for enrichment this is per sub-batch
  (`file_tokens + sum(chunk_tokens in that sub-batch)`), not per whole file.
- Retries/backoff on 429s stay with the `openai` SDK's built-in retry
  (`max_retries` on the client).
- Used by both `enrichment.py` and `embeddings.py`, so the throttle is
  shared across whichever calls are in flight at once, not tracked
  separately per module.

## Pipeline restructure

`ingest_repository` moves from one interleaved per-file loop to a
producer/consumer pipeline:

1. **Parse phase** — walk `list_source_files`, parse every file, collect
   `ParsedFile`s (chunks + source + imports) in memory. No LLM calls.
   Memory cost is source text for the whole repo, already on disk from the
   clone — negligible relative to the clone itself even at 100x scale.
   Feeds a repo-wide file queue.

2. **Enrichment workers** — a bounded pool (`ENRICHMENT_MAX_CONCURRENCY`)
   pulls files off that queue. Each worker enriches its file (one call, or
   several if split per the chunk cap above), validates the response(s),
   and pushes the file's now-enriched chunks onto a second, shared
   "ready to embed" queue.

3. **Embedding consumer** — drains the ready-to-embed queue, accumulating
   chunks **across files** until it has `EMBEDDING_BATCH_SIZE` (or the
   enrichment side signals it's fully drained, for a final partial flush),
   fires one `embed_texts` call for the accumulated batch, then upserts
   the whole batch. Only then are those files considered durably done.

This keeps embedding batches full regardless of individual file size
(a batch may span many small files), rather than the naive "embed+upsert
immediately after each file's enrichment" version, which would usually
submit far-under-filled embedding calls since most files won't have
`EMBEDDING_BATCH_SIZE` chunks on their own.

### Crash recovery — no disk queue needed

What needs to survive a crash is the record of *completed* work, not the
queue of pending work — re-parsing after a restart is cheap (local,
no LLM calls). That record already exists durably in Qdrant: deterministic
chunk IDs (`uuid5(file_path + symbol_name)`) plus the `content_hash`
already stored in each point's payload (`chunks.py:40`), with
`get_chunks_by_id` (`chunks.py:101`) already able to look them up.

- Before enqueuing a file for enrichment, check whether its chunks are
  already in Qdrant with matching `content_hash`; skip if so.
- Recovery granularity = one embedding-batch's worth of files (whatever
  was accumulated in the ready-to-embed queue at crash time), since that's
  the unit that becomes durable together in step 3. Bounded, not
  unbounded, and no new on-disk format to build or maintain.

## Embeddings: mostly unchanged, now fed by the consumer above

`embed_chunks`'s batching logic (`EMBEDDING_BATCH_SIZE` per call) stays;
what changes is *what feeds it* — a cross-file accumulation queue instead
of one file's chunks at a time. Embedding calls are gated through the same
`rate_limiter.py` as enrichment (not deferred) — one shared token budget
across both, since both draw from the same account-level RPM/TPM limits.
Embeddings were never the sequential-call bottleneck enrichment was, so
this is about correctness of the shared budget (not double-spending
capacity the enrichment side already accounted for) rather than fixing a
throughput problem on the embedding side itself.

## Module-by-module changes

- `rate_limiter.py` — new, per above.
- `enrichment.py` — `enrich_chunks` becomes async. New batched-call
  path: builds a structured-output prompt per file (or per sub-batch, if
  over `ENRICHMENT_MAX_CHUNKS_PER_CALL`), calls via `AsyncOpenAI` gated by
  the rate limiter, validates the response index set, retries missing
  indices as a scoped follow-up call. `_PROMPT_TEMPLATE` is replaced with a
  multi-chunk, structured-output version.
- `repository_ingester.py` — `ingest_repository` restructured into the
  parse → enrichment-workers → embedding-consumer pipeline above; becomes
  (or wraps) an `async def`. Adds the pre-enqueue Qdrant skip-check.
- `.env` additions (defaults below; user adds these to `.env` directly):

  | Var | Default | Rationale |
  |---|---|---|
  | `OPENAI_MAX_REQUESTS_PER_MINUTE` | `250` | 50% of cap, verified in dashboard. |
  | `OPENAI_MAX_TOKENS_PER_MINUTE` | `100000` | 50% of cap, verified in dashboard. |
  | `ENRICHMENT_MAX_CONCURRENCY` | `10` | Worker pool size — bounds how many files are in flight locally; independent of the two caps above (those gate actual API calls regardless of worker count). |
  | `ENRICHMENT_MAX_CHUNKS_PER_CALL` | `20` | Sized off an output-token budget: ~1-2 sentence context (~40-60 tokens) + JSON overhead per entry ≈ ~80-100 output tokens/chunk; 20 chunks ≈ ~2000 output tokens, comfortably under typical output limits with margin. |
  | `ENRICHMENT_MAX_CALL_RETRIES` | `2` | Whole-call retry budget when a call fails outright (SDK raises — malformed/non-conforming response, not just a missing index within an otherwise-valid response). After retries are exhausted, the file's remaining unenriched chunks are logged and surfaced as an ingestion-level error for that file rather than retried indefinitely. Separate from the missing-index retry in the Validation section above, which is scoped per-chunk, not whole-call. |
- `docs/architecture.md` §2.1 — update from "for each source file" to the
  producer/consumer pipeline description once implemented.

## Testing

- New `tests/test_enrichment.py` (doesn't exist today) mirroring
  `tests/test_embeddings.py`'s style: monkeypatch the OpenAI-call
  boundary, assert on: chunk-id-set validation (missing/extra ids logged
  and retried correctly), the per-file split at `ENRICHMENT_MAX_CHUNKS_PER_CALL`,
  and that split calls for one file stay sequential (not fanned out).
- New tests for `rate_limiter.py`: bucket refill math, blocking when over
  capacity, concurrent acquire ordering.
- New/extended tests for the `repository_ingester.py` pipeline: embedding
  batches span multiple files, partial final batch gets flushed, and the
  Qdrant skip-check actually skips a file whose chunks are already present
  with a matching `content_hash`.
