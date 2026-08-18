from contextlib import contextmanager
from datetime import datetime, timezone

from repo_metadata import get_repo_metadata, update_commit_sha, write_repo_metadata


class FakeCursor:
    """Stands in for a psycopg `Cursor`."""

    def __init__(self, row: tuple | None) -> None:
        self._row = row

    def fetchone(self) -> tuple | None:
        return self._row


class FakeConnection:
    """Stands in for a psycopg `Connection`; records every `execute` call."""

    def __init__(self, row: tuple | None = None) -> None:
        self.executed: list[tuple[str, tuple | None]] = []
        self.row = row

    def execute(self, sql: str, params: tuple | None = None) -> FakeCursor:
        self.executed.append((sql, params))
        return FakeCursor(self.row)


class FakePool:
    """Stands in for a psycopg `ConnectionPool` instead of hitting a real Postgres server."""

    def __init__(self, row: tuple | None = None) -> None:
        self.conn = FakeConnection(row)

    @contextmanager
    def connection(self):
        yield self.conn


def test_write_repo_metadata_upserts_the_given_url_and_sha() -> None:
    pool = FakePool()

    write_repo_metadata(pool, "https://github.com/example/repo", "abc123")

    sql, params = pool.conn.executed[0]
    assert "INSERT INTO repo_metadata" in sql
    assert "ON CONFLICT" in sql
    assert params == ("https://github.com/example/repo", "abc123")


def test_update_commit_sha_updates_only_the_sha() -> None:
    pool = FakePool()

    update_commit_sha(pool, "def456")

    sql, params = pool.conn.executed[0]
    assert "UPDATE repo_metadata" in sql
    assert params == ("def456",)


def test_get_repo_metadata_returns_none_when_no_row_exists() -> None:
    pool = FakePool(row=None)

    assert get_repo_metadata(pool) is None


def test_get_repo_metadata_returns_the_row_as_repo_metadata() -> None:
    updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pool = FakePool(row=("https://github.com/example/repo", "abc123", updated_at))

    result = get_repo_metadata(pool)

    assert result == {
        "github_url": "https://github.com/example/repo",
        "commit_sha": "abc123",
        "updated_at": updated_at,
    }
