import subprocess
from pathlib import Path, PurePosixPath

import pytest

import repo_diff
from repo_diff import diff_since, fetch_changed_files
from repository_clone import get_current_commit_sha


@pytest.fixture
def repo_with_history(tmp_path: Path) -> Path:
    """A real git repo with one baseline commit: a file to keep, one to modify, one to remove."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo_path, check=True)

    (repo_path / "keep.py").write_bytes(b"print('keep')\n")
    (repo_path / "modify.py").write_bytes(b"print('old')\n")
    (repo_path / "remove.py").write_bytes(b"print('gone soon')\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True, capture_output=True)
    return repo_path


def _commit_changes(repo_path: Path) -> None:
    (repo_path / "modify.py").write_bytes(b"print('new')\n")
    (repo_path / "added.py").write_bytes(b"print('added')\n")
    (repo_path / "remove.py").unlink()
    subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "second"], cwd=repo_path, check=True, capture_output=True)


def test_diff_since_reports_added_modified_and_deleted(repo_with_history: Path) -> None:
    old_sha = get_current_commit_sha(repo_with_history)
    _commit_changes(repo_with_history)
    new_sha = get_current_commit_sha(repo_with_history)

    changes = diff_since(repo_with_history, old_sha, new_sha)

    by_path = {change["path"]: change["status"] for change in changes}
    assert by_path == {
        PurePosixPath("modify.py"): "modified",
        PurePosixPath("added.py"): "added",
        PurePosixPath("remove.py"): "deleted",
    }


def test_diff_since_surfaces_a_rename_as_delete_plus_add(repo_with_history: Path) -> None:
    """--no-renames forces a rename to show as D+A, never an R status this pipeline can't parse."""
    old_sha = get_current_commit_sha(repo_with_history)
    (repo_with_history / "keep.py").rename(repo_with_history / "renamed.py")
    subprocess.run(["git", "add", "-A"], cwd=repo_with_history, check=True)
    subprocess.run(["git", "commit", "-m", "rename"], cwd=repo_with_history, check=True, capture_output=True)
    new_sha = get_current_commit_sha(repo_with_history)

    changes = diff_since(repo_with_history, old_sha, new_sha)

    by_path = {change["path"]: change["status"] for change in changes}
    assert by_path[PurePosixPath("keep.py")] == "deleted"
    assert by_path[PurePosixPath("renamed.py")] == "added"


def test_diff_since_raises_on_an_unrecognized_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A status this pipeline doesn't map fails loudly rather than being silently dropped."""

    class FakeResult:
        stdout = "C100\tcopied.py\n"

    monkeypatch.setattr(repo_diff.subprocess, "run", lambda *args, **kwargs: FakeResult())

    with pytest.raises(ValueError):
        diff_since(tmp_path, "a" * 40, "b" * 40)


def test_fetch_changed_files_returns_content_at_new_sha(repo_with_history: Path) -> None:
    old_sha = get_current_commit_sha(repo_with_history)
    _commit_changes(repo_with_history)
    new_sha = get_current_commit_sha(repo_with_history)

    files = fetch_changed_files(
        repo_with_history, new_sha, [PurePosixPath("modify.py"), PurePosixPath("added.py")]
    )

    by_path = dict(files)
    assert by_path[PurePosixPath("modify.py")] == b"print('new')\n"
    assert by_path[PurePosixPath("added.py")] == b"print('added')\n"


def test_fetch_changed_files_is_a_noop_for_an_empty_path_list(repo_with_history: Path) -> None:
    """Guards against `git archive` with no pathspec, which would archive the whole tree."""
    new_sha = get_current_commit_sha(repo_with_history)

    files = fetch_changed_files(repo_with_history, new_sha, [])

    assert files == []
