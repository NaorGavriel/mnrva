import os
import shutil
import stat
import subprocess
from pathlib import Path


class CloneError(RuntimeError):
    """Raised when `git clone` fails - bad URL, network failure, or a private repo needing auth."""


class UpdateError(RuntimeError):
    """Raised when updating an existing clone fails - `git fetch` or `git reset --hard`."""


def _rmtree(path: Path) -> None:
    """Delete a directory tree, clearing read-only attributes on failure and retrying.

    Git creates read-only files , which `shutil.rmtree` can't delete on Windows.
    """

    def _on_error(func, path, exc_info) -> None:
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(path, onexc=_on_error)


def clone_repository(github_url: str, dest_dir: Path) -> Path:
    """Clone `github_url` into `dest_dir`, wiping any existing directory there first.
    Any URL `git clone` accepts works.
    """
    if dest_dir.exists():
        _rmtree(dest_dir)
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


def clone_full_repository(github_url: str, dest_dir: Path) -> Path:
    """Full clone (no `--depth`/`--filter`) of `github_url` into `dest_dir`, wiping any existing directory there first.

    Used for the query agent's per-process clone (`docs/query_agent.md`
    §2.3/§2.8): full history so `update_repository` can reset to any commit
    offline, without depending on the remote staying reachable.
    """
    if dest_dir.exists():
        _rmtree(dest_dir)
    try:
        subprocess.run(
            ["git", "clone", github_url, str(dest_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise CloneError(f"failed to clone {github_url!r}: {e.stderr}") from e
    return dest_dir


def get_remote_head_sha(github_url: str) -> str:
    """Return origin's current HEAD commit sha via `git ls-remote`, no clone."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", github_url, "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise CloneError(f"failed to read HEAD of {github_url!r}: {e.stderr}") from e
    return result.stdout.split()[0]


def clone_for_diffing(github_url: str, dest_dir: Path) -> Path:
    """Partial, checkout-less clone of `github_url` into `dest_dir`.
    clones commit/tree objects only, nothing ever written outside `.git/`."""
    if dest_dir.exists():
        _rmtree(dest_dir)
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", github_url, str(dest_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise CloneError(f"failed to clone {github_url!r}: {e.stderr}") from e
    return dest_dir


def update_repository(repo_path: Path, commit_sha: str) -> None:
    """Bring an existing full clone at `repo_path` up to `commit_sha`.

    `git fetch origin` (the default refspec covers every branch, so nothing
    needs explicit tracking) followed by `git reset --hard <commit_sha>` -
    only files that actually changed get rewritten.
    """
    try:
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "reset", "--hard", commit_sha],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise UpdateError(f"failed to update {repo_path} to {commit_sha!r}: {e.stderr}") from e


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
    _rmtree(repo_path)


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
