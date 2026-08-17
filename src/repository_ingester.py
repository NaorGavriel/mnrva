from pathlib import Path

from qdrant_client import QdrantClient

from code_parser import parse_code_file
from db import COLLECTION_NAME, QDRANT_URL, ensure_collection, init_client, upsert_chunks
from embeddings import embed_chunks
from enrichment import enrich_chunks
from languages import get_language, is_code_file
from prose_parser import is_prose_file, parse_prose_file
from registry import GrammarRegistry, LanguageRegistry
from repository_clone import (
    clone_repository,
    delete_repository,
    get_current_commit_sha,
    prune_unwanted_files,
)
from repository_parser import list_source_files, list_unwanted_files

REPOSITORY_FILES_DIR = Path("repository_files")


def ingest_repository(
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

    repo_path = clone_repository(github_url, REPOSITORY_FILES_DIR)

    unwanted_files = list_unwanted_files(repo_path)
    prune_unwanted_files(repo_path, unwanted_files)

    commit_sha = get_current_commit_sha(repo_path)

    total_chunks = 0
    for relative_path in list_source_files(repo_path):
        if is_code_file(relative_path):
            language = get_language(relative_path)
            parsed = parse_code_file(repo_path / relative_path, language, registry, repo_root=repo_path)
        elif is_prose_file(relative_path):
            parsed = parse_prose_file(repo_path / relative_path, repo_root=repo_path)
        else:
            continue  # allowlisted but unroutable: neither a code nor prose extension

        enriched = enrich_chunks(parsed.chunks, parsed.source, parsed.imports)
        embedded = embed_chunks(enriched)
        upsert_chunks(client, COLLECTION_NAME, embedded)
        total_chunks += len(embedded)

    delete_repository(repo_path)
    print(f"upserted {total_chunks} chunks from {github_url} @ {commit_sha}")
    return commit_sha
