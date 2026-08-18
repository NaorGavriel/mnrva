from datetime import datetime
from typing import TypedDict

from psycopg_pool import ConnectionPool


class RepoMetadata(TypedDict):
    """The single tracked repo's github_url/commit_sha/updated_at, as read from Postgres."""

    github_url: str
    commit_sha: str
    updated_at: datetime


def write_repo_metadata(pool: ConnectionPool, github_url: str, commit_sha: str) -> None:
    """Write the repo-metadata row: insert it on first ingest, overwrite it on a re-ingest."""
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO repo_metadata (id, github_url, commit_sha, updated_at)
            VALUES (1, %s, %s, now())
            ON CONFLICT (id) DO UPDATE
            SET github_url = EXCLUDED.github_url,
                commit_sha = EXCLUDED.commit_sha,
                updated_at = EXCLUDED.updated_at
            """,
            (github_url, commit_sha),
        )


def update_commit_sha(pool: ConnectionPool, commit_sha: str) -> None:
    """Advance the tracked repo's commit_sha, e.g. after the refresh pipeline re-syncs the index."""
    with pool.connection() as conn:
        conn.execute(
            "UPDATE repo_metadata SET commit_sha = %s, updated_at = now() WHERE id = 1",
            (commit_sha,),
        )


def get_repo_metadata(pool: ConnectionPool) -> RepoMetadata | None:
    """Return the tracked repo's metadata, or None if it hasn't been ingested yet."""
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT github_url, commit_sha, updated_at FROM repo_metadata WHERE id = 1"
        ).fetchone()
    if row is None:
        return None
    github_url, commit_sha, updated_at = row
    return RepoMetadata(github_url=github_url, commit_sha=commit_sha, updated_at=updated_at)
