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


def prune_unwanted_files(repo_path: Path, wanted_files: list[Path]) -> list[Path]:
    """Delete every git-tracked file under `repo_path` that isn't in `wanted_files`.

    Also removes directories left empty by the deletions. `.git/` is never
    touched — only paths returned by `git ls-files` are ever considered for
    deletion.
    """
    tracked = _git_ls_files(repo_path)
    wanted = set(wanted_files)
    removed = [path for path in tracked if path not in wanted]
    for path in removed:
        (repo_path / path).unlink()
    _prune_empty_directories(repo_path)
    return removed


def delete_repository(repo_path: Path) -> None:
    """Delete the entire cloned repository at `repo_path`.

    Meant to be called by the orchestrator only after that run's chunks
    have been successfully embedded and upserted into Qdrant.
    """
    shutil.rmtree(repo_path)


def _git_ls_files(repo_path: Path) -> list[Path]:
    """Return every git-tracked file path under `repo_path`, relative to it."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


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
