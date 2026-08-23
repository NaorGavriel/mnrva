from pathlib import PurePosixPath

from qdrant_client import QdrantClient, models

from db.db_qdrant import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from embeddings import embed_text
from models import Chunk, ChunkSearchResult, chunk_retrieval_text


def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list[Chunk]) -> None:
    """Upsert every chunk's dense vector, sparse (BM25) vector, and payload.

    The sparse vector is generated server-side by Qdrant.
    """

    if not chunks:
        return
    points = [
        models.PointStruct(
            id=chunk.id,
            vector={
                DENSE_VECTOR_NAME: chunk.embedding,
                SPARSE_VECTOR_NAME: models.Document(
                    text=chunk_retrieval_text(chunk), model="Qdrant/bm25"
                ),
            },
            payload={
                "file_path": chunk.path.as_posix(),
                "language": chunk.language,
                "kind": chunk.kind,
                "class_name": chunk.class_name,
                "symbol_name": chunk.symbol_name,
                "start_byte": chunk.start_byte,
                "end_byte": chunk.end_byte,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "parent_id": chunk.parent_id,
                "raw_text": chunk.raw_text,
                "context_text": chunk.context_text,
                "content_hash": chunk.content_hash,
            },
        )
        for chunk in chunks
    ]
    client.upsert(collection_name=collection_name, points=points)


def search_chunks(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    top_k: int = 10,
    language: str | None = None,
) -> list[ChunkSearchResult]:
    """Hybrid dense + BM25 search over chunks, fused with Qdrant's native RRF.

    Optionally restricts results to a given `language` via a payload filter.
    """

    conditions = []
    if language is not None:
        conditions.append(models.FieldCondition(key="language", match=models.MatchValue(value=language)))

    query_filter = models.Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=embed_text(query_text), using=DENSE_VECTOR_NAME, filter=query_filter
            ),
            models.Prefetch(
                query=models.Document(text=query_text, model="Qdrant/bm25"),
                using=SPARSE_VECTOR_NAME,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )

    return [
        ChunkSearchResult(
            id=str(point.id),
            file_path=point.payload["file_path"],
            symbol_name=point.payload["symbol_name"],
            class_name=point.payload["class_name"],
            kind=point.payload["kind"],
            start_byte=point.payload["start_byte"],
            end_byte=point.payload["end_byte"],
            start_line=point.payload["start_line"],
            end_line=point.payload["end_line"],
            raw_text=point.payload["raw_text"],
            context_text=point.payload["context_text"],
            score=point.score,
        )
        for point in response.points
    ]


def get_chunks_by_id(client: QdrantClient, collection_name: str, chunk_ids: list[str]) -> list[Chunk]:
    """Batched point lookup by id - a direct fetch, not a similarity search.

    Ids that no longer exist (e.g. a chunk removed by a later refresh) are
    simply absent from the result rather than raising - `client.retrieve`
    already skips missing ids instead of erroring.
    """
    records = client.retrieve(collection_name=collection_name, ids=chunk_ids, with_payload=True)
    return [
        Chunk(
            id=str(record.id),
            content_hash=record.payload["content_hash"],
            path=PurePosixPath(record.payload["file_path"]),
            language=record.payload["language"],
            kind=record.payload["kind"],
            class_name=record.payload["class_name"],
            symbol_name=record.payload["symbol_name"],
            raw_text=record.payload["raw_text"],
            start_byte=record.payload["start_byte"],
            end_byte=record.payload["end_byte"],
            start_line=record.payload["start_line"],
            end_line=record.payload["end_line"],
            parent_id=record.payload["parent_id"],
            context_text=record.payload["context_text"],
        )
        for record in records
    ]
