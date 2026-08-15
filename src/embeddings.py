import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embedding_model = os.environ["EMBEDDING_MODEL"]


def embed_text(text: str) -> list[float]:
    response = _client.embeddings.create(model=embedding_model, input=text)
    return response.data[0].embedding
