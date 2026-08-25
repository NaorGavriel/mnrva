import os
from pathlib import PurePath

import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from models import Chunk
from rate_limiter import rate_limiter

load_dotenv()

_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
_enrichment_model = os.environ["ENRICHMENT_MODEL"]
ENRICHMENT_MAX_CHUNKS_PER_CALL = int(os.environ["ENRICHMENT_MAX_CHUNKS_PER_CALL"])
ENRICHMENT_MAX_CALL_RETRIES = int(os.environ["ENRICHMENT_MAX_CALL_RETRIES"])

try:
    _encoding = tiktoken.encoding_for_model(_enrichment_model)
except KeyError:
    _encoding = tiktoken.get_encoding("o200k_base")  # newer models unknown to tiktoken's lookup table

_PROMPT_TEMPLATE = """<file path="{path}">
imports:
{imports_block}

{source}
</file>

<chunks>
{chunks_block}
</chunks>

For each chunk above, give a short (1-2 sentence) context situating it \
within the file, for the purpose of improving search retrieval of the \
chunk. Return one entry per chunk index."""

_CHUNK_TEMPLATE = """[{index}] <chunk symbol="{symbol}">
{raw_text}
</chunk>"""


class ChunkContext(BaseModel):
    """One response entry: a chunk's local call-index plus its generated context."""

    index: int
    context: str


class EnrichmentResponse(BaseModel):
    """The structured-output shape for one enrichment call: one entry per requested chunk."""

    contexts: list[ChunkContext]


async def enrich_chunks(chunks: list[Chunk], source: str, imports: list[str]) -> list[Chunk]:
    """Populate `context_text` on every chunk from one file in place.

    Splits `chunks` into sub-batches of at most `ENRICHMENT_MAX_CHUNKS_PER_CALL`.
    """
    imports_block = "\n".join(imports)
    for start in range(0, len(chunks), ENRICHMENT_MAX_CHUNKS_PER_CALL):
        batch = chunks[start : start + ENRICHMENT_MAX_CHUNKS_PER_CALL]
        await _enrich_batch(batch, source, imports_block)
    return chunks


async def _enrich_batch(batch: list[Chunk], source: str, imports_block: str) -> None:
    """Enrich one sub-batch. Whatever's still missing context after a call -
    whether because the call failed outright or its response just didn't
    cover every index - is re-requested as a smaller, re-indexed call scoped
    to only what's left, up to `ENRICHMENT_MAX_CALL_RETRIES` times.
    """
    path = batch[0].path
    requested = batch

    for attempt in range(ENRICHMENT_MAX_CALL_RETRIES + 1):
        contexts, missing = await _try_generate_contexts(requested, source, imports_block, path)
        _apply_contexts(requested, contexts)
        if not missing:
            return
        if attempt < ENRICHMENT_MAX_CALL_RETRIES:
            print(f"enrichment: {len(missing)}/{len(batch)} chunks still missing context for {path}, retrying")
        requested = missing

    for chunk in requested:
        print(f"enrichment: {_chunk_label(chunk)} in {path} failed to enrich after retries")


async def _try_generate_contexts(
    chunks: list[Chunk], source: str, imports_block: str, path: PurePath
) -> tuple[dict[int, str], list[Chunk]]:
    """One enrichment call attempt for `chunks`. Returns `({}, chunks)` if the
    call fails outright (raises) - treated the same as a response that
    covered nothing.
    """
    prompt = _build_prompt(path, imports_block, source, chunks)
    estimated_tokens = _estimate_tokens(prompt)
    try:
        response = await generate_contexts(prompt, estimated_tokens)
        return _validate_response(response, chunks, path)
    except Exception as error:  # SDK raised: malformed/non-conforming response, network error, refusal, etc.
        print(f"enrichment: call failed for {path}: {error}")
        return {}, chunks


async def generate_contexts(prompt: str, estimated_tokens: int) -> EnrichmentResponse:
    """Call the enrichment model once with a batched prompt, gated by the
    shared rate limiter. The OpenAI-call boundary - monkeypatched in tests.
    """
    await rate_limiter.acquire(estimated_tokens)
    response = await _client.chat.completions.parse(
        model=_enrichment_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=EnrichmentResponse,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"enrichment call returned no parsed response: {response.choices[0].message.refusal}")
    return parsed


def _validate_response(
    response: EnrichmentResponse, chunks: list[Chunk], path: PurePath
) -> tuple[dict[int, str], list[Chunk]]:
    """The one place that checks a call's response against `chunks`:
    out-of-range or duplicate indices are logged and dropped rather than
    trusted; any chunk whose index never got a valid entry is returned in
    `missing`, for the caller to retry.
    """
    contexts: dict[int, str] = {}
    for entry in response.contexts:
        if not (0 <= entry.index < len(chunks)):
            print(f"enrichment: out-of-range index {entry.index} (expected 0-{len(chunks) - 1}) for {path}, dropping")
            continue
        if entry.index in contexts:
            print(f"enrichment: duplicate index {entry.index} for {path}, dropping")
            continue
        contexts[entry.index] = entry.context

    missing = [chunk for index, chunk in enumerate(chunks) if index not in contexts]
    return contexts, missing


def _apply_contexts(chunks: list[Chunk], contexts: dict[int, str]) -> None:
    """Set `context_text` on each chunk in `chunks` whose local (0-based) index is present in `contexts`."""
    for index, chunk in enumerate(chunks):
        if index in contexts:
            chunk.context_text = contexts[index]


def _build_prompt(path: PurePath, imports_block: str, source: str, chunks: list[Chunk]) -> str:
    """Render the batched enrichment prompt: file source/imports once, then
    every chunk in `chunks` addressed by its position in the list (0-based).
    """
    chunks_block = "\n".join(
        _CHUNK_TEMPLATE.format(index=index, symbol=_chunk_label(chunk), raw_text=chunk.raw_text)
        for index, chunk in enumerate(chunks)
    )
    return _PROMPT_TEMPLATE.format(path=path, imports_block=imports_block, source=source, chunks_block=chunks_block)


def _chunk_label(chunk: Chunk) -> str:
    """The chunk's display identity for prompts/logs: its symbol name, or its class name for a class chunk."""
    return chunk.symbol_name or chunk.class_name


def _estimate_tokens(prompt: str) -> int:
    """Token count for `prompt` via tiktoken, for gating the rate limiter."""
    return len(_encoding.encode(prompt))
