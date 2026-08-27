import os

import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from models import Chunk, chunk_retrieval_text
from rate_limiter import rate_limiter

load_dotenv()

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
embedding_model = os.environ["EMBEDDING_MODEL"]
EMBEDDING_BATCH_SIZE = int(os.environ["EMBEDDING_BATCH_SIZE"])  # conservative vs. the embeddings endpoint's ~2048-item/request cap

try:
    _encoding = tiktoken.encoding_for_model(embedding_model)
except KeyError:
    _encoding = tiktoken.get_encoding("o200k_base")  # newer models unknown to tiktoken's lookup table


def embed_text(text: str) -> list[float]:
    """Embed a single string via the configured OpenAI embedding model."""
    response = _client.embeddings.create(model=embedding_model, input=text)
    return response.data[0].embedding


async def aembed_text(text: str) -> list[float]:
    """Async twin of embed_text: embed a single string, gated by the shared rate limiter."""
    [embedding] = await embed_texts([text])
    return embedding


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings in one API call, gated by the shared rate limiter; order matches input order."""
    if not texts:
        return []
    estimated_tokens = sum(len(_encoding.encode(text)) for text in texts)
    await rate_limiter.acquire(estimated_tokens)
    response = await _async_client.embeddings.create(model=embedding_model, input=texts)
    return [item.embedding for item in response.data]


async def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Populate `embedding` on every chunk in place, batching requests to the embeddings endpoint."""
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        embeddings = await embed_texts([chunk_retrieval_text(chunk) for chunk in batch])
        for chunk, embedding in zip(batch, embeddings):
            chunk.embedding = embedding
    return chunks
