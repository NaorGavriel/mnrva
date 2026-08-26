import subprocess
from pathlib import Path

import pytest

import repository_parser
from repository_parser import (
    DENIED_FILENAMES,
    list_source_files,
    list_tracked_files,
    list_unwanted_files,
)


@pytest.fixture
def repo_with_files(tmp_path: Path) -> Path:
    """A real git repo with a mix of wanted, denylisted, and unsupported-extension files."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo_path, check=True)

    (repo_path / "main.py").write_text("print('hello')\n")
    (repo_path / "README.md").write_text("# readme\n")
    (repo_path / "package-lock.json").write_text("{}\n")
    (repo_path / "logo.png").write_bytes(b"\x89PNG\r\n")

    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    return repo_path


def test_list_tracked_files_returns_every_tracked_path(repo_with_files: Path) -> None:
    """list_tracked_files returns all four committed files, unfiltered."""
    tracked = list_tracked_files(repo_with_files)

    assert set(tracked) == {
        Path("main.py"),
        Path("README.md"),
        Path("package-lock.json"),
        Path("logo.png"),
    }


def test_list_source_files_keeps_only_wanted_files(repo_with_files: Path) -> None:
    """The .py and .md files pass; the lockfile (denylist) and .png (extension) don't."""
    wanted = list_source_files(repo_with_files)

    assert set(wanted) == {Path("main.py"), Path("README.md")}


def test_list_unwanted_files_excludes_the_lockfile_and_unsupported_extension(
    repo_with_files: Path,
) -> None:
    """The lockfile and the unsupported-extension file are exactly what's unwanted."""
    unwanted = list_unwanted_files(repo_with_files)

    assert set(unwanted) == {Path("package-lock.json"), Path("logo.png")}


def test_source_and_unwanted_files_partition_tracked_files(repo_with_files: Path) -> None:
    """list_source_files and list_unwanted_files share `_is_wanted`, so together they
    account for every tracked file exactly once - no overlap, no gaps."""
    tracked = set(list_tracked_files(repo_with_files))
    wanted = set(list_source_files(repo_with_files))
    unwanted = set(list_unwanted_files(repo_with_files))

    assert wanted & unwanted == set()
    assert wanted | unwanted == tracked


def test_list_source_files_excludes_oversized_files(
    repo_with_files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that would otherwise be wanted is excluded once it exceeds the size cutoff."""
    monkeypatch.setattr(repository_parser, "MAX_FILE_SIZE_BYTES", 5)

    wanted = list_source_files(repo_with_files)

    assert Path("main.py") not in wanted


@pytest.mark.parametrize("filename", sorted(DENIED_FILENAMES))
def test_is_denied_filename_matches_every_known_lockfile(filename: str) -> None:
    assert repository_parser._is_denied_filename(Path(filename))


def test_is_denied_filename_allows_ordinary_files() -> None:
    assert not repository_parser._is_denied_filename(Path("main.py"))


def test_passes_size_cutoff() -> None:
    assert repository_parser._passes_size_cutoff(1, max_bytes=10)
    assert not repository_parser._passes_size_cutoff(1, max_bytes=0)
