import os

from dotenv import load_dotenv
from openai import OpenAI

from models import Chunk

load_dotenv()

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_enrichment_model = os.environ["ENRICHMENT_MODEL"]

_PROMPT_TEMPLATE = """<file path="{path}">
imports:
{imports_block}

{source}
</file>

<chunk>
{raw_text}
</chunk>

Give a short (1-2 sentence) context situating this chunk within the file, \
for the purpose of improving search retrieval of the chunk. Answer with \
only the context and nothing else."""


def enrich_chunks(chunks: list[Chunk], source: str, imports: list[str]) -> list[Chunk]:
    """Populate `context_text` on every chunk from one file in place, via one LLM call per chunk.

    `chunks` must all belong to the same file — `source`/`imports` are
    shared across the whole batch.
    """
    imports_block = "\n".join(imports)
    for chunk in chunks:
        chunk.context_text = _generate_context(chunk, source, imports_block)
    return chunks


def _generate_context(chunk: Chunk, source: str, imports_block: str) -> str:
    """Call the LLM once to produce a short blurb situating `chunk` within its file."""
    prompt = _PROMPT_TEMPLATE.format(
        path=chunk.path,
        imports_block=imports_block,
        source=source,
        raw_text=chunk.raw_text,
    )
    response = _client.chat.completions.create(
        model=_enrichment_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
