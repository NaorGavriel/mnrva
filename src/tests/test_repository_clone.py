import subprocess
from pathlib import Path

import pytest

from repository_clone import (
    CloneError,
    UpdateError,
    clone_for_diffing,
    clone_full_repository,
    clone_repository,
    get_current_commit_sha,
    get_remote_head_sha,
    prune_unwanted_files,
    update_repository,
)

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


def test_clone_full_repository_creates_working_checkout(local_repo: Path, tmp_path: Path) -> None:
    """A full clone of a real repo lands at dest_dir with a working .git/ and its tracked files."""
    dest_dir = tmp_path / "cloned"

    result = clone_full_repository(str(local_repo), dest_dir)

    assert result == dest_dir
    assert (dest_dir / ".git").is_dir()
    assert (dest_dir / "hello.py").exists()


def test_clone_full_repository_wipes_existing_dest_dir(local_repo: Path, tmp_path: Path) -> None:
    """A pre-existing dest_dir (e.g. leftover from a prior run) is wiped before cloning."""
    dest_dir = tmp_path / "cloned"
    dest_dir.mkdir()
    stale_file = dest_dir / "stale.txt"
    stale_file.write_text("leftover from a previous run")

    clone_full_repository(str(local_repo), dest_dir)

    assert not stale_file.exists()
    assert (dest_dir / "hello.py").exists()


def test_clone_full_repository_raises_clone_error_on_invalid_source(tmp_path: Path) -> None:
    """A source git can't reach (bad URL/path) surfaces as CloneError, not a bare subprocess error."""
    dest_dir = tmp_path / "cloned"
    missing_source = tmp_path / "does_not_exist"

    with pytest.raises(CloneError):
        clone_full_repository(str(missing_source), dest_dir)


def test_clone_full_repository_keeps_full_history_unlike_the_shallow_clone(
    local_repo: Path, tmp_path: Path
) -> None:
    """Distinguishes clone_full_repository from clone_repository's --depth=1: no shallow marker, full log."""
    (local_repo / "second.py").write_text("print('second')\n")
    subprocess.run(["git", "add", "."], cwd=local_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "second commit"], cwd=local_repo, check=True, capture_output=True
    )
    dest_dir = tmp_path / "cloned"

    clone_full_repository(str(local_repo), dest_dir)

    assert not (dest_dir / ".git" / "shallow").exists()
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=dest_dir, check=True, capture_output=True, text=True
    )
    assert len(log.stdout.strip().splitlines()) == 2


def test_get_remote_head_sha_returns_the_current_head_commit(local_repo: Path) -> None:
    expected = get_current_commit_sha(local_repo)

    assert get_remote_head_sha(str(local_repo)) == expected


def test_get_remote_head_sha_raises_clone_error_on_invalid_source(tmp_path: Path) -> None:
    missing_source = tmp_path / "does_not_exist"

    with pytest.raises(CloneError):
        get_remote_head_sha(str(missing_source))


def test_clone_for_diffing_creates_a_checkout_less_clone(local_repo: Path, tmp_path: Path) -> None:
    """--no-checkout: .git/ exists but no working tree files are ever written."""
    dest_dir = tmp_path / "cloned"

    result = clone_for_diffing(str(local_repo), dest_dir)

    assert result == dest_dir
    assert (dest_dir / ".git").is_dir()
    assert not (dest_dir / "hello.py").exists()


def test_clone_for_diffing_wipes_existing_dest_dir(local_repo: Path, tmp_path: Path) -> None:
    """A pre-existing dest_dir (e.g. leftover from a prior run) is wiped before cloning."""
    dest_dir = tmp_path / "cloned"
    dest_dir.mkdir()
    stale_file = dest_dir / "stale.txt"
    stale_file.write_text("leftover from a previous run")

    clone_for_diffing(str(local_repo), dest_dir)

    assert not stale_file.exists()
    assert (dest_dir / ".git").is_dir()


def test_clone_for_diffing_raises_clone_error_on_invalid_source(tmp_path: Path) -> None:
    dest_dir = tmp_path / "cloned"
    missing_source = tmp_path / "does_not_exist"

    with pytest.raises(CloneError):
        clone_for_diffing(str(missing_source), dest_dir)


def test_update_repository_fetches_and_resets_to_a_newer_commit(local_repo: Path, tmp_path: Path) -> None:
    """A commit made upstream after the clone is pulled in and checked out."""
    dest_dir = tmp_path / "cloned"
    clone_full_repository(str(local_repo), dest_dir)
    (local_repo / "second.py").write_text("print('second')\n")
    subprocess.run(["git", "add", "."], cwd=local_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "second commit"], cwd=local_repo, check=True, capture_output=True
    )
    new_sha = get_current_commit_sha(local_repo)

    update_repository(dest_dir, new_sha)

    assert get_current_commit_sha(dest_dir) == new_sha
    assert (dest_dir / "second.py").exists()


def test_update_repository_removes_files_deleted_upstream(local_repo: Path, tmp_path: Path) -> None:
    """`--hard` reset rewrites the working tree, so a file removed upstream disappears locally too."""
    dest_dir = tmp_path / "cloned"
    clone_full_repository(str(local_repo), dest_dir)
    (local_repo / "hello.py").unlink()
    subprocess.run(["git", "add", "-A"], cwd=local_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "remove hello.py"], cwd=local_repo, check=True, capture_output=True
    )
    new_sha = get_current_commit_sha(local_repo)

    update_repository(dest_dir, new_sha)

    assert not (dest_dir / "hello.py").exists()


def test_update_repository_can_reset_to_an_earlier_commit(local_repo: Path, tmp_path: Path) -> None:
    """update_repository isn't fast-forward-only - it can roll back to any known sha."""
    initial_sha = get_current_commit_sha(local_repo)
    (local_repo / "second.py").write_text("print('second')\n")
    subprocess.run(["git", "add", "."], cwd=local_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "second commit"], cwd=local_repo, check=True, capture_output=True
    )
    dest_dir = tmp_path / "cloned"
    clone_full_repository(str(local_repo), dest_dir)

    update_repository(dest_dir, initial_sha)

    assert get_current_commit_sha(dest_dir) == initial_sha
    assert not (dest_dir / "second.py").exists()


def test_update_repository_raises_update_error_for_an_unknown_commit(
    local_repo: Path, tmp_path: Path
) -> None:
    """A sha that doesn't exist anywhere reachable surfaces as UpdateError, not a bare subprocess error."""
    dest_dir = tmp_path / "cloned"
    clone_full_repository(str(local_repo), dest_dir)

    with pytest.raises(UpdateError):
        update_repository(dest_dir, "0" * 40)


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
