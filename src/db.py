from qdrant_client import QdrantClient, models

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small
COLLECTION_NAME = "code_chunks"
QDRANT_URL = "http://localhost:6333"


def init_client(path: str | None = None, url: str | None = None) -> QdrantClient:
    if path is not None:
        return QdrantClient(path=path)
    if url is not None:
        return QdrantClient(url=url)
    raise ValueError("init_client requires either path or url")


def ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=EMBEDDING_DIM, distance=models.Distance.COSINE
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            ),
        },
    )
