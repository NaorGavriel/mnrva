import asyncio
import os
import time

from qdrant_client import AsyncQdrantClient, QdrantClient, models

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small
COLLECTION_NAME = "code_chunks"
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")


def init_client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    """Connect to Qdrant: `path=` for an embedded local instance, `url=` for a real server."""
    if url is not None:
        return QdrantClient(url=url, api_key=api_key)
    raise ValueError("init_client requires url")


def init_async_client(url: str | None = None, api_key: str | None = None) -> AsyncQdrantClient:
    """Async twin of init_client."""
    if url is not None:
        return AsyncQdrantClient(url=url, api_key=api_key)
    raise ValueError("init_async_client requires url")


async def wait_until_ready(client: AsyncQdrantClient, timeout: float = 30.0, interval: float = 1.0) -> None:
    """Poll Qdrant with `get_collections()` every `interval` seconds until it responds or `timeout` elapses."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            await client.get_collections()
            return
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(interval)
    raise ConnectionError(f"Qdrant at {QDRANT_URL} not reachable after {timeout}s") from last_error


def ensure_collection(client: QdrantClient, name: str) -> None:
    """Create the `name` collection (dense + BM25 sparse vectors) if it doesn't already exist."""
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
