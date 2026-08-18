from qdrant_client import QdrantClient, models

from db_qdrant import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from models import Chunk, chunk_retrieval_text


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
