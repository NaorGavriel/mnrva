from pathlib import Path

from psycopg_pool import ConnectionPool

from repo_metadata import get_repo_metadata
from repository_clone import clone_full_repository, get_current_commit_sha, update_repository


def sync_clone(pool: ConnectionPool, repo_path: Path) -> Path:
    """Bring the local clone at `repo_path` in line with the ingested commit tracked in Postgres.

    No clone yet -> full clone. Clone present but on a different commit ->
    fetch + reset to the tracked commit_sha.
    Runs at process startup and at the start of every conversation (`docs/query_agent.md` §2.3).
    """
    metadata = get_repo_metadata(pool)
    if metadata is None:
        raise RuntimeError("no repo has been ingested yet - repo_metadata is empty")

    if not (repo_path / ".git").is_dir():
        clone_full_repository(metadata["github_url"], repo_path)

    if get_current_commit_sha(repo_path) != metadata["commit_sha"]:
        update_repository(repo_path, metadata["commit_sha"])

    return repo_path
