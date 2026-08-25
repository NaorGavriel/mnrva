import asyncio
import re
from pathlib import PurePosixPath

import pytest

import enrichment
from enrichment import ChunkContext, EnrichmentResponse, enrich_chunks
from models import Chunk


def _make_chunk(symbol_name: str) -> Chunk:
    return Chunk(
        id=f"id-{symbol_name}",
        content_hash="hash",
        path=PurePosixPath("main.py"),
        language="python",
        kind="function",
        class_name="",
        symbol_name=symbol_name,
        raw_text=f"def {symbol_name}(): pass",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=1,
        parent_id=None,
    )


async def test_enrich_chunks_populates_context_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every chunk gets context_text from the response entry at its call-local index."""

    async def fake_generate_contexts(prompt: str, estimated_tokens: int) -> EnrichmentResponse:
        return EnrichmentResponse(
            contexts=[ChunkContext(index=0, context="first"), ChunkContext(index=1, context="second")]
        )

    monkeypatch.setattr(enrichment, "generate_contexts", fake_generate_contexts)
    chunks = [_make_chunk("greet"), _make_chunk("farewell")]

    enriched = await enrich_chunks(chunks, source="source", imports=[])

    assert [chunk.context_text for chunk in enriched] == ["first", "second"]


async def test_enrich_chunks_retries_a_missing_index_as_a_scoped_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunk absent from the first call's response gets one smaller, re-indexed follow-up call."""
    prompts: list[str] = []

    async def fake_generate_contexts(prompt: str, estimated_tokens: int) -> EnrichmentResponse:
        prompts.append(prompt)
        if len(prompts) == 1:
            return EnrichmentResponse(contexts=[ChunkContext(index=0, context="first")])
        return EnrichmentResponse(contexts=[ChunkContext(index=0, context="retried")])

    monkeypatch.setattr(enrichment, "generate_contexts", fake_generate_contexts)
    chunks = [_make_chunk("greet"), _make_chunk("farewell")]

    enriched = await enrich_chunks(chunks, source="source", imports=[])

    assert enriched[0].context_text == "first"
    assert enriched[1].context_text == "retried"
    assert len(prompts) == 2
    assert "farewell" in prompts[1] and "greet" not in prompts[1]


async def test_enrich_chunks_drops_invalid_indices_and_gives_up_if_retry_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-range indices are dropped, not trusted; a chunk still missing once
    retries are exhausted stays unenriched rather than retried forever."""
    monkeypatch.setattr(enrichment, "ENRICHMENT_MAX_CALL_RETRIES", 1)
    call_count = 0

    async def fake_generate_contexts(prompt: str, estimated_tokens: int) -> EnrichmentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return EnrichmentResponse(
                contexts=[ChunkContext(index=0, context="ok"), ChunkContext(index=5, context="out of range")]
            )
        return EnrichmentResponse(contexts=[])

    monkeypatch.setattr(enrichment, "generate_contexts", fake_generate_contexts)
    chunks = [_make_chunk("greet"), _make_chunk("farewell")]

    enriched = await enrich_chunks(chunks, source="source", imports=[])

    assert enriched[0].context_text == "ok"
    assert enriched[1].context_text is None
    assert call_count == 2


async def test_enrich_chunks_splits_at_the_configured_chunk_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file with more chunks than ENRICHMENT_MAX_CHUNKS_PER_CALL is split across multiple calls."""
    monkeypatch.setattr(enrichment, "ENRICHMENT_MAX_CHUNKS_PER_CALL", 2)
    batch_sizes: list[int] = []

    async def fake_generate_contexts(prompt: str, estimated_tokens: int) -> EnrichmentResponse:
        n = len(re.findall(r"^\[\d+\] <chunk", prompt, re.MULTILINE))
        batch_sizes.append(n)
        return EnrichmentResponse(contexts=[ChunkContext(index=i, context="c") for i in range(n)])

    monkeypatch.setattr(enrichment, "generate_contexts", fake_generate_contexts)
    chunks = [_make_chunk(f"fn{i}") for i in range(5)]

    enriched = await enrich_chunks(chunks, source="source", imports=[])

    assert batch_sizes == [2, 2, 1]
    assert all(chunk.context_text == "c" for chunk in enriched)


async def test_enrich_chunks_issues_split_calls_sequentially_not_fanned_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sub-batch calls for one file never overlap in flight - issued one at a time."""
    monkeypatch.setattr(enrichment, "ENRICHMENT_MAX_CHUNKS_PER_CALL", 1)
    in_flight = 0
    max_in_flight = 0

    async def fake_generate_contexts(prompt: str, estimated_tokens: int) -> EnrichmentResponse:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return EnrichmentResponse(contexts=[ChunkContext(index=0, context="c")])

    monkeypatch.setattr(enrichment, "generate_contexts", fake_generate_contexts)
    chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]

    await enrich_chunks(chunks, source="source", imports=[])

    assert max_in_flight == 1


async def test_enrich_chunks_gives_up_after_max_call_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call that keeps failing outright is retried up to ENRICHMENT_MAX_CALL_RETRIES times,
    then the chunk is left unenriched rather than retried indefinitely."""
    monkeypatch.setattr(enrichment, "ENRICHMENT_MAX_CALL_RETRIES", 1)
    call_count = 0

    async def fake_generate_contexts(prompt: str, estimated_tokens: int) -> EnrichmentResponse:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(enrichment, "generate_contexts", fake_generate_contexts)
    chunks = [_make_chunk("greet")]

    enriched = await enrich_chunks(chunks, source="source", imports=[])

    assert enriched[0].context_text is None
    assert call_count == 2
