from pathlib import Path

import pytest

import embeddings
from embeddings import embed_chunks
from models import Chunk


def _make_chunk(context_text: str | None = None) -> Chunk:
    return Chunk(
        id="chunk-id",
        content_hash="hash",
        path=Path("main.py"),
        language="python",
        kind="function",
        class_name="",
        symbol_name="greet",
        raw_text="def greet(): pass",
        start_byte=0,
        end_byte=18,
        start_line=1,
        end_line=1,
        parent_id=None,
        context_text=context_text,
    )


async def test_embed_chunks_populates_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every chunk in the result has `embedding` set from embed_texts's return value, in order."""

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)

    embedded = await embed_chunks([_make_chunk(), _make_chunk(context_text="context")])

    assert [chunk.embedding for chunk in embedded] == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]


async def test_embed_chunks_embeds_context_plus_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk with context_text embeds context + raw_text, matching chunk_retrieval_text."""
    seen_batches: list[list[str]] = []

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        seen_batches.append(texts)
        return [[] for _ in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)

    await embed_chunks([_make_chunk(context_text="this greets someone")])

    assert seen_batches == [["this greets someone\n\ndef greet(): pass"]]


async def test_embed_chunks_embeds_raw_text_alone_without_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunk with no context_text yet embeds raw_text alone."""
    seen_batches: list[list[str]] = []

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        seen_batches.append(texts)
        return [[] for _ in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)

    await embed_chunks([_make_chunk()])

    assert seen_batches == [["def greet(): pass"]]


async def test_embed_chunks_mutates_chunks_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_chunks sets `embedding` directly on the given chunks and returns the same objects."""

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[9.0] for _ in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)
    chunk = _make_chunk()

    embedded = await embed_chunks([chunk])

    assert chunk.embedding == [9.0]
    assert embedded[0] is chunk


async def test_embed_chunks_preserves_order_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output has the same chunks, in the same order, as input."""

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)
    chunks = [_make_chunk(), _make_chunk(context_text="c")]

    embedded = await embed_chunks(chunks)

    assert len(embedded) == 2
    assert [chunk.raw_text for chunk in embedded] == [chunk.raw_text for chunk in chunks]


async def test_embed_chunks_batches_requests_at_embedding_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """More chunks than EMBEDDING_BATCH_SIZE are split across multiple embed_texts calls."""
    monkeypatch.setattr(embeddings, "EMBEDDING_BATCH_SIZE", 2)
    seen_batch_sizes: list[int] = []

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        seen_batch_sizes.append(len(texts))
        return [[0.0] for _ in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)
    chunks = [_make_chunk(), _make_chunk(), _make_chunk()]

    await embed_chunks(chunks)

    assert seen_batch_sizes == [2, 1]
