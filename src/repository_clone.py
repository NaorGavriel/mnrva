import shutil
import subprocess
from pathlib import Path


class CloneError(RuntimeError):
    """Raised when `git clone` fails - bad URL, network failure, or a private repo needing auth."""


def clone_repository(github_url: str, dest_dir: Path) -> Path:
    """Clone `github_url` into `dest_dir`, wiping any existing directory there first.
    Any URL `git clone` accepts works.
    """
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", github_url, str(dest_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise CloneError(f"failed to clone {github_url!r}: {e.stderr}") from e
    return dest_dir


def get_current_commit_sha(repo_path: Path) -> str:
    """Return the current HEAD commit sha of the git checkout at `repo_path`."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def prune_unwanted_files(repo_path: Path, unwanted_files: list[Path]) -> list[Path]:
    """Delete every path in `unwanted_files` from `repo_path`.

    Paths under `.git/` are always skipped, even if given — a defensive
    guard against a caller passing something it shouldn't.
    """
    removed = []
    for path in unwanted_files:
        if ".git" in path.parts:
            continue
        full_path = repo_path / path
        if full_path.exists():
            full_path.unlink()
            removed.append(path)
    _prune_empty_directories(repo_path)
    return removed


def delete_repository(repo_path: Path) -> None:
    """Delete the entire cloned repository at `repo_path`.

    Meant to be called by the orchestrator only after that run's chunks
    have been successfully embedded and upserted into Qdrant.
    """
    shutil.rmtree(repo_path)


def _prune_empty_directories(repo_path: Path) -> None:
    """Recursively delete directories under `repo_path` left empty by pruning.

    Skips `.git` entirely.
    `repo_path` itself is never removed, even if it ends up empty.
    """

    def _clean(directory: Path) -> None:
        for child in directory.iterdir():
            if child.is_dir() and child.name != ".git":
                _clean(child)
        if directory != repo_path and not any(directory.iterdir()):
            directory.rmdir()

    _clean(repo_path)
