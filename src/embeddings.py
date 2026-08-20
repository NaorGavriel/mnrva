import os

from dotenv import load_dotenv
from openai import OpenAI

from models import Chunk, chunk_retrieval_text

load_dotenv()

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embedding_model = os.environ["EMBEDDING_MODEL"]
EMBEDDING_BATCH_SIZE = int(os.environ["EMBEDDING_BATCH_SIZE"])  # conservative vs. the embeddings endpoint's ~2048-item/request cap


def embed_text(text: str) -> list[float]:
    """Embed a single string via the configured OpenAI embedding model."""
    response = _client.embeddings.create(model=embedding_model, input=text)
    return response.data[0].embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings in one API call; order matches input order."""
    if not texts:
        return []
    response = _client.embeddings.create(model=embedding_model, input=texts)
    return [item.embedding for item in response.data]


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Populate `embedding` on every chunk in place, batching requests to the embeddings endpoint."""
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        for chunk, embedding in zip(batch, embed_texts([chunk_retrieval_text(chunk) for chunk in batch])):
            chunk.embedding = embedding
    return chunks
