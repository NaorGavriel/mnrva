from pathlib import Path, PurePath
from typing import TypedDict

from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient

from chunks import delete_chunks_by_path
from code_parser import parse_code_bytes
from db.db_qdrant import COLLECTION_NAME
from languages import get_language, is_code_file
from models import ParsedFile
from prose_parser import is_prose_file, parse_prose_bytes
from registry import GrammarRegistry, LanguageRegistry
from repo_diff import diff_since, fetch_changed_files
from repo_metadata import get_repo_metadata, update_commit_sha
from repository_clone import clone_for_diffing, delete_repository, get_remote_head_sha
from repository_ingester import enrich_embed_and_upsert
from repository_parser import MAX_FILE_SIZE_BYTES, passes_allowlist, passes_size_cutoff


class RefreshError(RuntimeError):
    """Raised when the refresh pipeline can't proceed - repo_metadata has no row yet."""


class RefreshResult(TypedDict):
    """What one refresh run changed, for the caller (cron/CLI/webhook) to log."""

    added: int
    modified: int
    deleted: int
    old_sha: str
    new_sha: str


async def sync_repository(
    pool: ConnectionPool,
    client: QdrantClient,
    repo_path: Path,
    registry: GrammarRegistry | None = None,
) -> RefreshResult:
    """Bring Qdrant and repo_metadata's commit_sha up to origin's current HEAD.

    A no-op if the tracked commit already matches origin. 
    On failure the run aborts before `update_commit_sha`.
    """
    registry = registry or LanguageRegistry()

    metadata = get_repo_metadata(pool)
    if metadata is None:
        raise RefreshError("repo_metadata has no row - nothing has been ingested yet")

    old_sha = metadata["commit_sha"]
    github_url = metadata["github_url"]
    new_sha = get_remote_head_sha(github_url)

    if old_sha == new_sha:
        return RefreshResult(added=0, modified=0, deleted=0, old_sha=old_sha, new_sha=new_sha)

    clone_for_diffing(github_url, repo_path)

    changes = diff_since(repo_path, old_sha, new_sha)

    changed_paths: list[PurePath] = []
    deleted_paths: list[PurePath] = []
    status_by_changed_path: dict[PurePath, str] = {}
    for change in changes:
        if change["status"] == "deleted":
            deleted_paths.append(change["path"])
        elif passes_allowlist(change["path"]):
            changed_paths.append(change["path"])
            status_by_changed_path[change["path"]] = change["status"]
        else:
            deleted_paths.append(change["path"]) # A changed-status path failing the allowlist is treated as a deletion.

    deleted_count = len(deleted_paths)  # snapshot before changed paths are appended below for delete-then-replace

    parsed_files: list[ParsedFile] = []
    for path, content in fetch_changed_files(repo_path, new_sha, changed_paths):
        deleted_paths.append(path)  # delete-then-replace, before any upsert
        if not passes_size_cutoff(len(content), MAX_FILE_SIZE_BYTES):
            continue  # oversized post-fetch: chunks purged above, never re-parsed
        if is_code_file(path):
            parsed_files.append(parse_code_bytes(content, path, get_language(path), registry))
        elif is_prose_file(path):
            parsed_files.append(parse_prose_bytes(content.decode("utf-8"), path))

    for path in deleted_paths:
        delete_chunks_by_path(client, COLLECTION_NAME, path)

    await enrich_embed_and_upsert(client, parsed_files)

    update_commit_sha(pool, new_sha)
    delete_repository(repo_path)

    return RefreshResult(
        added=sum(1 for f in parsed_files if status_by_changed_path.get(f.path) == "added"),
        modified=sum(1 for f in parsed_files if status_by_changed_path.get(f.path) == "modified"),
        deleted=deleted_count,
        old_sha=old_sha,
        new_sha=new_sha,
    )
