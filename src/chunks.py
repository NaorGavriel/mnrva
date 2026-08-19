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
    kind: str | None = None,
) -> list[ChunkSearchResult]:
    """Hybrid dense + BM25 search over chunks, fused with Qdrant's native RRF.

    Optionally restricts results to a given `language` and/or `kind` via a
    payload filter.
    """

    conditions = []
    if language is not None:
        conditions.append(models.FieldCondition(key="language", match=models.MatchValue(value=language)))
    if kind is not None:
        conditions.append(models.FieldCondition(key="kind", match=models.MatchValue(value=kind)))
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
            file_path=point.payload["file_path"],
            symbol_name=point.payload["symbol_name"],
            class_name=point.payload["class_name"],
            kind=point.payload["kind"],
            start_byte=point.payload["start_byte"],
            end_byte=point.payload["end_byte"],
            raw_text=point.payload["raw_text"],
            context_text=point.payload["context_text"],
            score=point.score,
        )
        for point in response.points
    ]
