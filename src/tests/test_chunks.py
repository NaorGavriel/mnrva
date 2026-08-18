from pathlib import PurePosixPath

from qdrant_client import models as qdrant_models

from chunks import upsert_chunks
from db_qdrant import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from models import Chunk, chunk_retrieval_text


class FakeQdrantClient:
    """Records upsert calls instead of hitting a real Qdrant server."""

    def __init__(self) -> None:
        self.upsert_calls: list[dict] = []

    def upsert(self, collection_name: str, points: list) -> None:
        self.upsert_calls.append({"collection_name": collection_name, "points": points})


def _make_chunk(chunk_id: str = "11111111-1111-1111-1111-111111111111", **overrides) -> Chunk:
    defaults = dict(
        id=chunk_id,
        content_hash="hash",
        path=PurePosixPath("src/main.py"),
        language="python",
        kind="function",
        class_name="",
        symbol_name="greet",
        raw_text="def greet(): pass",
        start_byte=0,
        end_byte=18,
        parent_id=None,
        context_text="greets someone",
        embedding=[0.1, 0.2, 0.3],
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def test_upsert_chunks_calls_upsert_with_the_given_collection_name() -> None:
    client = FakeQdrantClient()

    upsert_chunks(client, "code_chunks", [_make_chunk()])

    assert len(client.upsert_calls) == 1
    assert client.upsert_calls[0]["collection_name"] == "code_chunks"


def test_upsert_chunks_builds_one_point_per_chunk_with_matching_ids() -> None:
    client = FakeQdrantClient()
    chunks = [
        _make_chunk("11111111-1111-1111-1111-111111111111"),
        _make_chunk("22222222-2222-2222-2222-222222222222"),
    ]

    upsert_chunks(client, "code_chunks", chunks)

    points = client.upsert_calls[0]["points"]
    assert [point.id for point in points] == [chunk.id for chunk in chunks]


def test_upsert_chunks_sets_dense_vector_from_chunk_embedding() -> None:
    client = FakeQdrantClient()
    chunk = _make_chunk(embedding=[1.0, 2.0, 3.0])

    upsert_chunks(client, "code_chunks", [chunk])

    point = client.upsert_calls[0]["points"][0]
    assert point.vector[DENSE_VECTOR_NAME] == [1.0, 2.0, 3.0]


def test_upsert_chunks_sets_sparse_vector_as_a_bm25_document() -> None:
    """Sparse vector is a Document Qdrant computes BM25 from, not a hand-rolled sparse vector."""
    client = FakeQdrantClient()
    chunk = _make_chunk()

    upsert_chunks(client, "code_chunks", [chunk])

    point = client.upsert_calls[0]["points"][0]
    sparse_vector = point.vector[SPARSE_VECTOR_NAME]
    assert isinstance(sparse_vector, qdrant_models.Document)
    assert sparse_vector.model == "Qdrant/bm25"
    assert sparse_vector.text == chunk_retrieval_text(chunk)


def test_upsert_chunks_builds_payload_from_chunk_fields() -> None:
    client = FakeQdrantClient()
    chunk = _make_chunk(
        path=PurePosixPath("src/nested/main.py"),
        language="python",
        kind="function",
        class_name="Greeter",
        symbol_name="greet",
        start_byte=10,
        end_byte=40,
        parent_id="parent-id",
        content_hash="abc123",
        raw_text="def greet_method(): pass",
        context_text='a method that greets'
    )

    upsert_chunks(client, "code_chunks", [chunk])

    point = client.upsert_calls[0]["points"][0]
    assert point.payload == {
        "file_path": "src/nested/main.py",
        "language": "python",
        "kind": "function",
        "class_name": "Greeter",
        "symbol_name": "greet",
        "start_byte": 10,
        "end_byte": 40,
        "parent_id": "parent-id",
        "content_hash": "abc123",
        "raw_text":"def greet_method(): pass",
        "context_text":'a method that greets',
    }


def test_upsert_chunks_is_a_noop_for_empty_chunks() -> None:
    """Qdrant rejects an upsert with zero points; an empty chunk list must not call it."""
    client = FakeQdrantClient()

    upsert_chunks(client, "code_chunks", [])

    assert client.upsert_calls == []
