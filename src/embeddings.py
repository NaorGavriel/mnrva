import os

from dotenv import load_dotenv
from openai import OpenAI

from models import Chunk, chunk_retrieval_text

load_dotenv()

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embedding_model = os.environ["EMBEDDING_MODEL"]


def embed_text(text: str) -> list[float]:
    """Embed a single string via the configured OpenAI embedding model."""
    response = _client.embeddings.create(model=embedding_model, input=text)
    return response.data[0].embedding


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Populate `embedding` on every chunk in place, one `embed_text` call each."""
    for chunk in chunks:
        chunk.embedding = embed_text(chunk_retrieval_text(chunk))
    return chunks
