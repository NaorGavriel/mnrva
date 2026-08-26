import subprocess
from pathlib import Path

from languages import LANGUAGE_CONFIG
from prose_parser import PROSE_EXTENSIONS

DENIED_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "composer.lock",
    "Gemfile.lock",
    "go.sum",
    "mix.lock",
    "flake.lock",
}

MAX_FILE_SIZE_BYTES = 1_000_000


def list_tracked_files(repo_path: Path) -> list[Path]:
    """Return every git-tracked file path under `repo_path`, relative to it."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def list_source_files(repo_path: Path) -> list[Path]:
    """Return the git-tracked files ingestion should actually process.

    `git ls-files`, filtered through the allowlist, a filename denylist for files (e.g. lockfiles) 
    that would otherwise pass the extension allowlist, and a file-size cutoff.
    """
    return [
        path
        for path in list_tracked_files(repo_path)
        if _is_wanted(repo_path, path)
    ]


def list_unwanted_files(repo_path: Path) -> list[Path]:
    """Return the git-tracked files `list_source_files` excludes."""
    return [
        path
        for path in list_tracked_files(repo_path)
        if not _is_wanted(repo_path, path)
    ]


def _is_wanted(repo_path: Path, path: Path) -> bool:
    """Whether `path` passes the extension allowlist, filename denylist, and size cutoff."""
    return passes_allowlist(path) and _passes_size_cutoff(
        (repo_path / path).stat().st_size, MAX_FILE_SIZE_BYTES
    )


def passes_allowlist(path: Path) -> bool:
    """Whether `path` passes the extension allowlist and filename denylist, on path alone."""
    allowed_extensions = set(LANGUAGE_CONFIG) | PROSE_EXTENSIONS
    return path.suffix in allowed_extensions and not _is_denied_filename(path)


def _is_denied_filename(path: Path) -> bool:
    """Whether `path`'s filename is a lockfile (or similar) excluded regardless of extension."""
    return path.name in DENIED_FILENAMES


def _passes_size_cutoff(length: int, max_bytes: int) -> bool:
    """Whether a file of `length` bytes is at or under `max_bytes`."""
    return length <= max_bytes
