import subprocess
from pathlib import Path

import pytest

from query_agent import sync_clone as sync_clone_module
from query_agent.sync_clone import sync_clone
from repository_clone import clone_full_repository, get_current_commit_sha


def _fail(*args, **kwargs):
    raise AssertionError("should not be called")


@pytest.fixture
def local_repo(tmp_path: Path) -> Path:
    """A minimal real git repository on disk, standing in for the origin `sync_clone` clones from."""
    repo_path = tmp_path / "source_repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo_path, check=True)
    (repo_path / "hello.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"], cwd=repo_path, check=True, capture_output=True
    )
    return repo_path


def _commit_second_file(repo_path: Path) -> None:
    (repo_path / "second.py").write_text("print('second')\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "second commit"], cwd=repo_path, check=True, capture_output=True
    )


def test_sync_clone_raises_when_no_repo_has_been_ingested(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sync_clone_module, "get_repo_metadata", lambda pool: None)

    with pytest.raises(RuntimeError):
        sync_clone(pool=object(), repo_path=tmp_path / "clone")


def test_sync_clone_cold_starts_a_missing_clone(monkeypatch, local_repo: Path, tmp_path: Path) -> None:
    sha = get_current_commit_sha(local_repo)
    monkeypatch.setattr(
        sync_clone_module,
        "get_repo_metadata",
        lambda pool: {"github_url": str(local_repo), "commit_sha": sha, "updated_at": None},
    )
    dest_dir = tmp_path / "clone"

    result = sync_clone(pool=object(), repo_path=dest_dir)

    assert result == dest_dir
    assert (dest_dir / ".git").is_dir()
    assert get_current_commit_sha(dest_dir) == sha


def test_sync_clone_cold_starts_and_resets_to_the_tracked_commit_even_if_origin_has_moved_on(
    monkeypatch, local_repo: Path, tmp_path: Path
) -> None:
    """A cold clone lands on origin's current HEAD by default, which may be ahead of what was indexed."""
    tracked_sha = get_current_commit_sha(local_repo)
    _commit_second_file(local_repo)
    assert get_current_commit_sha(local_repo) != tracked_sha
    monkeypatch.setattr(
        sync_clone_module,
        "get_repo_metadata",
        lambda pool: {"github_url": str(local_repo), "commit_sha": tracked_sha, "updated_at": None},
    )
    dest_dir = tmp_path / "clone"

    sync_clone(pool=object(), repo_path=dest_dir)

    assert get_current_commit_sha(dest_dir) == tracked_sha
    assert not (dest_dir / "second.py").exists()


def test_sync_clone_updates_an_existing_clone_on_a_mismatched_commit(
    monkeypatch, local_repo: Path, tmp_path: Path
) -> None:
    dest_dir = tmp_path / "clone"
    clone_full_repository(str(local_repo), dest_dir)
    _commit_second_file(local_repo)
    new_sha = get_current_commit_sha(local_repo)
    monkeypatch.setattr(
        sync_clone_module,
        "get_repo_metadata",
        lambda pool: {"github_url": str(local_repo), "commit_sha": new_sha, "updated_at": None},
    )

    sync_clone(pool=object(), repo_path=dest_dir)

    assert get_current_commit_sha(dest_dir) == new_sha
    assert (dest_dir / "second.py").exists()


def test_sync_clone_is_a_noop_when_already_on_the_tracked_commit(
    monkeypatch, local_repo: Path, tmp_path: Path
) -> None:
    dest_dir = tmp_path / "clone"
    clone_full_repository(str(local_repo), dest_dir)
    sha = get_current_commit_sha(dest_dir)
    monkeypatch.setattr(
        sync_clone_module,
        "get_repo_metadata",
        lambda pool: {"github_url": str(local_repo), "commit_sha": sha, "updated_at": None},
    )
    monkeypatch.setattr(sync_clone_module, "clone_full_repository", _fail)
    monkeypatch.setattr(sync_clone_module, "update_repository", _fail)

    result = sync_clone(pool=object(), repo_path=dest_dir)

    assert result == dest_dir
