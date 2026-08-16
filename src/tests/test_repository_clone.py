import subprocess
from pathlib import Path

import pytest

from repository_clone import CloneError, clone_repository, prune_unwanted_files

REMOTE_REPO_URL = "https://github.com/NaorGavriel/book-select.git"


@pytest.fixture
def local_repo(tmp_path: Path) -> Path:
    """A minimal real git repository on disk, for clone tests that shouldn't need the network."""
    repo_path = tmp_path / "source_repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo_path, check=True)
    (repo_path / "hello.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    return repo_path


def test_clone_repository_creates_working_checkout(local_repo: Path, tmp_path: Path) -> None:
    """A clone of a real repo lands at dest_dir with a working .git/ and its tracked files."""
    dest_dir = tmp_path / "cloned"

    result = clone_repository(str(local_repo), dest_dir)

    assert result == dest_dir
    assert (dest_dir / ".git").is_dir()
    assert (dest_dir / "hello.py").exists()


def test_clone_repository_wipes_existing_dest_dir(local_repo: Path, tmp_path: Path) -> None:
    """A pre-existing dest_dir (e.g. leftover from a prior run) is wiped before cloning."""
    dest_dir = tmp_path / "cloned"
    dest_dir.mkdir()
    stale_file = dest_dir / "stale.txt"
    stale_file.write_text("leftover from a previous run")

    clone_repository(str(local_repo), dest_dir)

    assert not stale_file.exists()
    assert (dest_dir / "hello.py").exists()


def test_clone_repository_raises_clone_error_on_invalid_source(tmp_path: Path) -> None:
    """A source git can't reach (bad URL/path) surfaces as CloneError, not a bare subprocess error."""
    dest_dir = tmp_path / "cloned"
    missing_source = tmp_path / "does_not_exist"

    with pytest.raises(CloneError):
        clone_repository(str(missing_source), dest_dir)


@pytest.mark.skipif(not REMOTE_REPO_URL, reason="set REMOTE_REPO_URL to run this network test")
def test_clone_repository_against_real_remote(tmp_path: Path) -> None:
    """End-to-end smoke test against a real remote, to catch anything the local fixture can't."""
    dest_dir = tmp_path / "cloned"

    result = clone_repository(REMOTE_REPO_URL, dest_dir)

    assert result == dest_dir
    assert (dest_dir / ".git").is_dir()
    assert any(dest_dir.iterdir())


def test_prune_unwanted_files_deletes_only_the_given_paths(tmp_path: Path) -> None:
    """Only the paths passed in are removed; everything else on disk is untouched."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "keep.py").write_text("kept\n")
    (repo_path / "remove.json").write_text("{}\n")

    removed = prune_unwanted_files(repo_path, [Path("remove.json")])

    assert removed == [Path("remove.json")]
    assert not (repo_path / "remove.json").exists()
    assert (repo_path / "keep.py").exists()


def test_prune_unwanted_files_removes_now_empty_directories(tmp_path: Path) -> None:
    """A directory left empty after its only file is pruned gets removed too."""
    repo_path = tmp_path / "repo"
    nested_dir = repo_path / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "remove.json").write_text("{}\n")
    (repo_path / "keep.py").write_text("kept\n")

    prune_unwanted_files(repo_path, [Path("nested/remove.json")])

    assert not nested_dir.exists()
    assert (repo_path / "keep.py").exists()


def test_prune_unwanted_files_keeps_non_empty_directories(tmp_path: Path) -> None:
    """A directory with a surviving file is left in place after pruning."""
    repo_path = tmp_path / "repo"
    nested_dir = repo_path / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "remove.json").write_text("{}\n")
    (nested_dir / "keep.py").write_text("kept\n")

    prune_unwanted_files(repo_path, [Path("nested/remove.json")])

    assert nested_dir.exists()
    assert (nested_dir / "keep.py").exists()


def test_prune_unwanted_files_never_touches_git_directory(tmp_path: Path) -> None:
    """.git/ is left alone even if it's (incorrectly) passed as unwanted."""
    repo_path = tmp_path / "repo"
    git_dir = repo_path / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "config").write_text("fake git config\n")

    removed = prune_unwanted_files(repo_path, [Path(".git/config")])

    assert removed == []
    assert (git_dir / "config").exists()


def test_prune_unwanted_files_ignores_paths_that_dont_exist(tmp_path: Path) -> None:
    """A path that's already gone (e.g. a stale entry) is skipped, not an error."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    removed = prune_unwanted_files(repo_path, [Path("does_not_exist.py")])

    assert removed == []
