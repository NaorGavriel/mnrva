import subprocess
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import pytest

import refresh_sync
from models import ParsedFile
from refresh_sync import RefreshError, sync_repository
from registry import LanguageRegistry
from repository_clone import get_current_commit_sha


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


@pytest.fixture
def tracked_repo(tmp_path: Path) -> Path:
    """A real git repo standing in for 'origin', with one baseline commit."""
    repo_path = tmp_path / "origin"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo_path, check=True)

    (repo_path / "main.py").write_text("def greet():\n    return 'hi'\n")
    (repo_path / "keep.py").write_text("print('keep')\n")
    (repo_path / "package-lock.json").write_text("{}\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True, capture_output=True)
    return repo_path


def _commit(repo_path: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo_path, check=True, capture_output=True)


def _patch_downstream(monkeypatch: pytest.MonkeyPatch) -> tuple[list[PurePosixPath], list[list[PurePosixPath]]]:
    """Fake out everything past the real git plumbing: chunk deletion and enrich/embed/upsert."""
    deleted_paths: list[PurePosixPath] = []
    upserted_paths: list[list[PurePosixPath]] = []

    monkeypatch.setattr(
        refresh_sync, "delete_chunks_by_path", lambda client, collection, path: deleted_paths.append(path)
    )

    async def fake_enrich_embed_and_upsert(client, parsed_files: list[ParsedFile]) -> int:
        upserted_paths.append([f.path for f in parsed_files])
        return sum(len(f.chunks) for f in parsed_files)

    monkeypatch.setattr(refresh_sync, "enrich_embed_and_upsert", fake_enrich_embed_and_upsert)
    return deleted_paths, upserted_paths


async def test_sync_repository_raises_when_nothing_has_been_ingested(tmp_path: Path) -> None:
    pool = FakePool(row=None)

    with pytest.raises(RefreshError):
        await sync_repository(pool, client=None, repo_path=tmp_path / "scratch")


async def test_sync_repository_is_a_noop_when_tracked_sha_matches_origin(
    tracked_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No clone at all when nothing changed - checked by making clone_for_diffing fail if called."""
    sha = get_current_commit_sha(tracked_repo)
    pool = FakePool(row=(str(tracked_repo), sha, None))
    monkeypatch.setattr(
        refresh_sync,
        "clone_for_diffing",
        lambda *args, **kwargs: pytest.fail("clone_for_diffing should not be called when nothing changed"),
    )

    result = await sync_repository(pool, client=None, repo_path=tmp_path / "scratch")

    assert result == {"added": 0, "modified": 0, "deleted": 0, "old_sha": sha, "new_sha": sha}


async def test_sync_repository_reindexes_added_and_modified_files(
    tracked_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_sha = get_current_commit_sha(tracked_repo)
    deleted_paths, upserted_paths = _patch_downstream(monkeypatch)
    (tracked_repo / "main.py").write_text("def greet():\n    return 'hello'\n")
    (tracked_repo / "added.py").write_text("def other():\n    return 1\n")
    _commit(tracked_repo, "modify and add")
    new_sha = get_current_commit_sha(tracked_repo)
    repo_path = tmp_path / "scratch"
    pool = FakePool(row=(str(tracked_repo), old_sha, None))

    result = await sync_repository(pool, client=None, repo_path=repo_path, registry=LanguageRegistry())

    assert result["old_sha"] == old_sha
    assert result["new_sha"] == new_sha
    assert result["added"] == 1
    assert result["modified"] == 1
    reindexed_paths = {path for batch in upserted_paths for path in batch}
    assert reindexed_paths == {PurePosixPath("main.py"), PurePosixPath("added.py")}
    assert set(deleted_paths) >= {PurePosixPath("main.py"), PurePosixPath("added.py")}


async def test_sync_repository_deletes_chunks_for_a_removed_file(
    tracked_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_sha = get_current_commit_sha(tracked_repo)
    deleted_paths, upserted_paths = _patch_downstream(monkeypatch)
    (tracked_repo / "keep.py").unlink()
    _commit(tracked_repo, "remove keep.py")
    pool = FakePool(row=(str(tracked_repo), old_sha, None))

    result = await sync_repository(pool, client=None, repo_path=tmp_path / "scratch", registry=LanguageRegistry())

    assert PurePosixPath("keep.py") in deleted_paths
    assert result["deleted"] == 1
    assert upserted_paths == [[]]


async def test_sync_repository_reclassifies_an_allowlist_rejected_path_as_a_deletion(
    tracked_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A modified lockfile (denylisted) still gets its stale chunks purged, but is never re-fetched/parsed."""
    old_sha = get_current_commit_sha(tracked_repo)
    deleted_paths, upserted_paths = _patch_downstream(monkeypatch)
    (tracked_repo / "package-lock.json").write_text('{"changed": true}\n')
    _commit(tracked_repo, "modify lockfile")
    pool = FakePool(row=(str(tracked_repo), old_sha, None))

    await sync_repository(pool, client=None, repo_path=tmp_path / "scratch", registry=LanguageRegistry())

    assert PurePosixPath("package-lock.json") in deleted_paths
    reindexed_paths = {path for batch in upserted_paths for path in batch}
    assert PurePosixPath("package-lock.json") not in reindexed_paths


async def test_sync_repository_skips_reindexing_an_oversized_file_but_still_deletes_it(
    tracked_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_sha = get_current_commit_sha(tracked_repo)
    deleted_paths, upserted_paths = _patch_downstream(monkeypatch)
    monkeypatch.setattr(refresh_sync, "MAX_FILE_SIZE_BYTES", 5)
    (tracked_repo / "main.py").write_text("def greet():\n    return 'this is over five bytes'\n")
    _commit(tracked_repo, "grow main.py past the cutoff")
    pool = FakePool(row=(str(tracked_repo), old_sha, None))

    await sync_repository(pool, client=None, repo_path=tmp_path / "scratch", registry=LanguageRegistry())

    assert PurePosixPath("main.py") in deleted_paths
    reindexed_paths = {path for batch in upserted_paths for path in batch}
    assert PurePosixPath("main.py") not in reindexed_paths
