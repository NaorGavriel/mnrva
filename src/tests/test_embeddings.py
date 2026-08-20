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


def test_embed_chunks_populates_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every chunk in the result has `embedding` set from embed_text's return value."""
    monkeypatch.setattr(embeddings, "embed_text", lambda text: [1.0, 2.0, 3.0])

    embedded = embed_chunks([_make_chunk(), _make_chunk(context_text="context")])

    assert [chunk.embedding for chunk in embedded] == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]


def test_embed_chunks_embeds_context_plus_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk with context_text embeds context + raw_text, matching chunk_retrieval_text."""
    seen_inputs: list[str] = []
    monkeypatch.setattr(embeddings, "embed_text", lambda text: seen_inputs.append(text) or [])

    embed_chunks([_make_chunk(context_text="this greets someone")])

    assert seen_inputs == ["this greets someone\n\ndef greet(): pass"]


def test_embed_chunks_embeds_raw_text_alone_without_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunk with no context_text yet embeds raw_text alone."""
    seen_inputs: list[str] = []
    monkeypatch.setattr(embeddings, "embed_text", lambda text: seen_inputs.append(text) or [])

    embed_chunks([_make_chunk()])

    assert seen_inputs == ["def greet(): pass"]


def test_embed_chunks_mutates_chunks_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_chunks sets `embedding` directly on the given chunks and returns the same objects."""
    monkeypatch.setattr(embeddings, "embed_text", lambda text: [9.0])
    chunk = _make_chunk()

    embedded = embed_chunks([chunk])

    assert chunk.embedding == [9.0]
    assert embedded[0] is chunk


def test_embed_chunks_preserves_order_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output has the same chunks, in the same order, as input."""
    monkeypatch.setattr(embeddings, "embed_text", lambda text: [0.0])
    chunks = [_make_chunk(), _make_chunk(context_text="c")]

    embedded = embed_chunks(chunks)

    assert len(embedded) == 2
    assert [chunk.raw_text for chunk in embedded] == [chunk.raw_text for chunk in chunks]
