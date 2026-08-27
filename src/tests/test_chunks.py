from pathlib import PurePosixPath

from qdrant_client import models as qdrant_models
from dotenv import load_dotenv

load_dotenv()

import chunks
from chunks import delete_chunks_by_path, get_chunks_by_id, search_chunks, upsert_chunks
from db.db_qdrant import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from models import Chunk, chunk_retrieval_text


class FakeQdrantClient:
    """Records upsert/query_points/retrieve calls instead of hitting a real Qdrant server."""

    def __init__(
        self,
        query_response: "FakeQueryResponse | None" = None,
        retrieve_response: "list[FakeRecord] | None" = None,
    ) -> None:
        self.upsert_calls: list[dict] = []
        self.query_points_calls: list[dict] = []
        self.retrieve_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self._query_response = query_response if query_response is not None else FakeQueryResponse([])
        self._retrieve_response = retrieve_response if retrieve_response is not None else []

    def upsert(self, collection_name: str, points: list) -> None:
        self.upsert_calls.append({"collection_name": collection_name, "points": points})

    def delete(self, collection_name: str, points_selector) -> None:
        self.delete_calls.append({"collection_name": collection_name, "points_selector": points_selector})

    def query_points(self, **kwargs) -> "FakeQueryResponse":
        self.query_points_calls.append(kwargs)
        return self._query_response

    def retrieve(self, **kwargs) -> "list[FakeRecord]":
        self.retrieve_calls.append(kwargs)
        return self._retrieve_response


class FakePoint:
    """Stands in for a Qdrant `ScoredPoint`."""

    def __init__(self, payload: dict, score: float, id: str = "11111111-1111-1111-1111-111111111111") -> None:
        self.payload = payload
        self.score = score
        self.id = id


class FakeQueryResponse:
    """Stands in for a Qdrant `QueryResponse`."""

    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class FakeRecord:
    """Stands in for a Qdrant `Record`, as returned by `retrieve`."""

    def __init__(self, payload: dict, id: str = "11111111-1111-1111-1111-111111111111") -> None:
        self.payload = payload
        self.id = id


def _make_point(score: float = 0.75, **payload_overrides) -> FakePoint:
    payload = dict(
        file_path="src/main.py",
        symbol_name="greet",
        class_name="",
        kind="function",
        start_byte=0,
        end_byte=18,
        start_line=1,
        end_line=1,
        raw_text="def greet(): pass",
        context_text="greets someone",
        language="python",
        parent_id=None,
        content_hash="hash",
    )
    payload.update(payload_overrides)
    return FakePoint(payload, score)


def _make_record(chunk_id: str = "11111111-1111-1111-1111-111111111111", **payload_overrides) -> FakeRecord:
    payload = dict(
        file_path="src/main.py",
        symbol_name="greet",
        class_name="",
        kind="function",
        start_byte=0,
        end_byte=18,
        start_line=1,
        end_line=1,
        raw_text="def greet(): pass",
        context_text="greets someone",
        language="python",
        parent_id=None,
        content_hash="hash",
    )
    payload.update(payload_overrides)
    return FakeRecord(payload, id=chunk_id)


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
        start_line=1,
        end_line=1,
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
        start_line=2,
        end_line=5,
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
        "start_line": 2,
        "end_line": 5,
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


def test_delete_chunks_by_path_deletes_by_the_file_path_payload_field() -> None:
    client = FakeQdrantClient()

    delete_chunks_by_path(client, "code_chunks", PurePosixPath("src/nested/main.py"))

    assert len(client.delete_calls) == 1
    call = client.delete_calls[0]
    assert call["collection_name"] == "code_chunks"
    condition = call["points_selector"].filter.must[0]
    assert condition.key == "file_path"
    assert condition.match.value == "src/nested/main.py"


def test_search_chunks_calls_query_points_with_the_given_collection_name(monkeypatch) -> None:
    client = FakeQdrantClient()
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])

    search_chunks(client, "code_chunks", "how does auth work")

    assert len(client.query_points_calls) == 1
    assert client.query_points_calls[0]["collection_name"] == "code_chunks"


def test_search_chunks_embeds_the_query_text_for_the_dense_prefetch(monkeypatch) -> None:
    client = FakeQdrantClient()
    monkeypatch.setattr(chunks, "embed_text", lambda text: [9.0, 9.0] if text == "how does auth work" else None)

    search_chunks(client, "code_chunks", "how does auth work")

    prefetches = client.query_points_calls[0]["prefetch"]
    dense_prefetch = next(p for p in prefetches if p.using == DENSE_VECTOR_NAME)
    assert dense_prefetch.query == [9.0, 9.0]


def test_search_chunks_uses_a_bm25_document_for_the_sparse_prefetch(monkeypatch) -> None:
    """Sparse side is a Document Qdrant computes BM25 from, not a hand-rolled sparse vector."""
    client = FakeQdrantClient()
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])

    search_chunks(client, "code_chunks", "how does auth work")

    prefetches = client.query_points_calls[0]["prefetch"]
    sparse_prefetch = next(p for p in prefetches if p.using == SPARSE_VECTOR_NAME)
    assert isinstance(sparse_prefetch.query, qdrant_models.Document)
    assert sparse_prefetch.query.model == "Qdrant/bm25"
    assert sparse_prefetch.query.text == "how does auth work"


def test_search_chunks_fuses_with_rrf_and_passes_top_k_as_limit(monkeypatch) -> None:
    client = FakeQdrantClient()
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])

    search_chunks(client, "code_chunks", "how does auth work", top_k=5)

    call = client.query_points_calls[0]
    assert isinstance(call["query"], qdrant_models.FusionQuery)
    assert call["query"].fusion == qdrant_models.Fusion.RRF
    assert call["limit"] == 5


def test_search_chunks_defaults_top_k_to_ten(monkeypatch) -> None:
    client = FakeQdrantClient()
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])

    search_chunks(client, "code_chunks", "how does auth work")

    assert client.query_points_calls[0]["limit"] == 10


def test_search_chunks_without_language_leaves_prefetches_unfiltered(monkeypatch) -> None:
    client = FakeQdrantClient()
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])

    search_chunks(client, "code_chunks", "how does auth work")

    prefetches = client.query_points_calls[0]["prefetch"]
    assert all(prefetch.filter is None for prefetch in prefetches)


def test_search_chunks_filters_both_prefetches_by_language(monkeypatch) -> None:
    """Filtering happens inside the prefetches so RRF fuses over already-filtered hits."""
    client = FakeQdrantClient()
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])

    search_chunks(client, "code_chunks", "how does auth work", language="python")

    prefetches = client.query_points_calls[0]["prefetch"]
    for prefetch in prefetches:
        assert prefetch.filter is not None
        conditions = prefetch.filter.must
        assert any(c.key == "language" and c.match.value == "python" for c in conditions)


def test_search_chunks_returns_chunk_search_results_from_hit_payload_and_score(monkeypatch) -> None:
    client = FakeQdrantClient(query_response=FakeQueryResponse([_make_point(score=0.42)]))
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])

    results = search_chunks(client, "code_chunks", "how does auth work")

    assert results == [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "file_path": "src/main.py",
            "symbol_name": "greet",
            "class_name": "",
            "kind": "function",
            "start_byte": 0,
            "end_byte": 18,
            "start_line": 1,
            "end_line": 1,
            "raw_text": "def greet(): pass",
            "context_text": "greets someone",
            "score": 0.42,
        }
    ]


def test_search_chunks_is_empty_list_when_there_are_no_hits(monkeypatch) -> None:
    client = FakeQdrantClient(query_response=FakeQueryResponse([]))
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])

    results = search_chunks(client, "code_chunks", "how does auth work")

    assert results == []


def test_get_chunks_by_id_calls_retrieve_with_the_given_collection_name_and_ids() -> None:
    client = FakeQdrantClient()

    get_chunks_by_id(client, "code_chunks", ["11111111-1111-1111-1111-111111111111"])

    assert len(client.retrieve_calls) == 1
    assert client.retrieve_calls[0]["collection_name"] == "code_chunks"
    assert client.retrieve_calls[0]["ids"] == ["11111111-1111-1111-1111-111111111111"]


def test_get_chunks_by_id_builds_chunks_from_record_id_and_payload() -> None:
    client = FakeQdrantClient(
        retrieve_response=[_make_record("11111111-1111-1111-1111-111111111111", raw_text="def greet(): pass")]
    )

    results = get_chunks_by_id(client, "code_chunks", ["11111111-1111-1111-1111-111111111111"])

    assert results == [
        Chunk(
            id="11111111-1111-1111-1111-111111111111",
            content_hash="hash",
            path=PurePosixPath("src/main.py"),
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
            context_text="greets someone",
        )
    ]


def test_get_chunks_by_id_omits_ids_not_returned_by_retrieve() -> None:
    """`retrieve` already skips ids Qdrant doesn't have - a deleted chunk isn't an error."""
    client = FakeQdrantClient(retrieve_response=[])

    results = get_chunks_by_id(client, "code_chunks", ["99999999-9999-9999-9999-999999999999"])

    assert results == []
