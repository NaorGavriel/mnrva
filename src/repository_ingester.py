import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from chunks import get_chunks_by_id, upsert_chunks
from code_parser import parse_code_file
from db.db_postgres import ensure_repo_metadata_table, init_pool
from db.db_qdrant import COLLECTION_NAME, QDRANT_URL, ensure_collection, init_client
from embeddings import EMBEDDING_BATCH_SIZE, embed_chunks
from enrichment import enrich_chunks
from languages import get_language, is_code_file
from models import Chunk, ParsedFile
from prose_parser import is_prose_file, parse_prose_file
from registry import GrammarRegistry, LanguageRegistry
from repository_clone import (
    clone_repository,
    delete_repository,
    get_current_commit_sha,
    prune_unwanted_files,
)
from repository_parser import list_source_files, list_unwanted_files

load_dotenv()

REPOSITORY_FILES_DIR = Path("repository_files")
ENRICHMENT_MAX_CONCURRENCY = int(os.environ["ENRICHMENT_MAX_CONCURRENCY"])


async def ingest_repository(
    github_url: str,
    registry: GrammarRegistry | None = None,
    client: QdrantClient | None = None,
) -> str:
    """Clone `github_url`, chunk/enrich/embed every wanted code and prose file, and upsert into Qdrant.

    Returns the commit sha that was ingested. Deletes the local scratch
    clone (`repository_files/`) on success; leaves it on disk if anything
    raises.
    """
    registry = registry or LanguageRegistry()
    client = client or init_client(url=QDRANT_URL)
    ensure_collection(client, COLLECTION_NAME)

    pool = init_pool()
    ensure_repo_metadata_table(pool=pool)

    repo_path = clone_repository(github_url, REPOSITORY_FILES_DIR)

    unwanted_files = list_unwanted_files(repo_path)
    prune_unwanted_files(repo_path, unwanted_files)

    commit_sha = get_current_commit_sha(repo_path)

    parsed_files = parse_repository_files(repo_path, registry)
    total_chunks = await _enrich_embed_and_upsert(client, parsed_files)

    delete_repository(repo_path)
    print(f"upserted {total_chunks} chunks from {github_url} @ {commit_sha}")
    return commit_sha


def parse_repository_files(repo_path: Path, registry: GrammarRegistry) -> list[ParsedFile]:
    """Parse every wanted file under `repo_path` into a `ParsedFile`.

    Routes each file to tree-sitter (code) or the prose splitter; a file
    that's allowlisted but unroutable to either is skipped. This is the
    parse phase: it only reads and chunks files, feeding the enrichment
    pipeline that follows.
    """
    parsed_files: list[ParsedFile] = []
    for relative_path in list_source_files(repo_path):
        if is_code_file(relative_path):
            language = get_language(relative_path)
            parsed = parse_code_file(repo_path / relative_path, language, registry, repo_root=repo_path)
        elif is_prose_file(relative_path):
            parsed = parse_prose_file(repo_path / relative_path, repo_root=repo_path)
        else:
            continue  # allowlisted but unroutable: neither a code nor prose extension
        parsed_files.append(parsed)
    return parsed_files


async def _enrich_embed_and_upsert(client: QdrantClient, parsed_files: list[ParsedFile]) -> int:
    """Run the producer/consumer pipeline over `parsed_files`.
    """
    file_queue: asyncio.Queue[ParsedFile] = asyncio.Queue()
    for parsed in parsed_files:
        if _needs_enrichment(client, parsed):
            file_queue.put_nowait(parsed)

    ready_queue: asyncio.Queue[list[Chunk] | None] = asyncio.Queue()
    consumer = asyncio.create_task(_embedding_consumer(client, ready_queue))

    worker_count = min(ENRICHMENT_MAX_CONCURRENCY, file_queue.qsize())
    workers = [asyncio.create_task(_enrichment_worker(file_queue, ready_queue)) for _ in range(worker_count)]
    await asyncio.gather(*workers)

    await ready_queue.put(None)  # signals the consumer that no more files are coming
    return await consumer


async def _enrichment_worker(
    file_queue: "asyncio.Queue[ParsedFile]", ready_queue: "asyncio.Queue[list[Chunk] | None]"
) -> None:
    """Pull files off `file_queue` until it's empty, enriching each and
    pushing its chunks onto `ready_queue` for the embedding consumer."""
    while True:
        try:
            parsed = file_queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        enriched = await enrich_chunks(parsed.chunks, parsed.source, parsed.imports)
        if enriched:
            await ready_queue.put(enriched)


async def _embedding_consumer(client: QdrantClient, ready_queue: "asyncio.Queue[list[Chunk] | None]") -> int:
    """Accumulate enriched chunks across files until there's a full
    `EMBEDDING_BATCH_SIZE` batch, embedding and upserting each as it fills.
    flushes whatever's left as one final partial batch once a `None`
    sentinel signals every enrichment worker is done. Returns the total
    number of chunks upserted.
    """
    accumulator: list[Chunk] = []
    total_upserted = 0
    while True:
        item = await ready_queue.get()
        if item is None:
            break
        accumulator.extend(item)
        while len(accumulator) >= EMBEDDING_BATCH_SIZE:
            batch, accumulator = accumulator[:EMBEDDING_BATCH_SIZE], accumulator[EMBEDDING_BATCH_SIZE:]
            total_upserted += await _embed_and_upsert(client, batch)

    if accumulator:
        total_upserted += await _embed_and_upsert(client, accumulator)
    return total_upserted


async def _embed_and_upsert(client: QdrantClient, chunks: list[Chunk]) -> int:
    """Embed one batch of chunks and upsert it into Qdrant, returning the batch size."""
    embedded = await embed_chunks(chunks)
    upsert_chunks(client, COLLECTION_NAME, embedded)
    return len(embedded)


def _needs_enrichment(client: QdrantClient, parsed: ParsedFile) -> bool:
    """Whether `parsed` still needs enrichment - false only if every one of
    its chunks already exists in Qdrant with a matching `content_hash`.
    """
    if not parsed.chunks:
        return False
    existing_hashes = {
        chunk.id: chunk.content_hash
        for chunk in get_chunks_by_id(client, COLLECTION_NAME, [chunk.id for chunk in parsed.chunks])
    }
    return not all(existing_hashes.get(chunk.id) == chunk.content_hash for chunk in parsed.chunks)
