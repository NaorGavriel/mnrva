import io
import subprocess
import tarfile
from pathlib import Path, PurePath, PurePosixPath
from typing import Literal, TypedDict


class FileChange(TypedDict):
    """One line of `git diff --name-status`: what changed and its identity, never its content."""

    status: Literal["added", "modified", "deleted"]
    path: PurePath


_STATUS_CODES: dict[str, Literal["added", "modified", "deleted"]] = {
    "A": "added",
    "M": "modified",
    "D": "deleted",
}


def diff_since(repo_path: Path, old_sha: str, new_sha: str) -> list[FileChange]:
    """List every file changed between `old_sha` and `new_sha`, identity only."""
    
    result = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", old_sha, new_sha],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    changes: list[FileChange] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        code, path_str = line.split("\t", 1)
        status = _STATUS_CODES.get(code)
        if status is None:
            raise ValueError(f"unrecognized git diff status {code!r} for {path_str!r}")
        changes.append(FileChange(status=status, path=PurePosixPath(path_str)))
    return changes


def fetch_changed_files(
    repo_path: Path, new_sha: str, changed_paths: list[PurePath]
) -> list[tuple[PurePath, bytes]]:
    """Batch-fetch every path in `changed_paths`'s content at `new_sha`, one `git archive` call.

    Never called with `deleted_paths` - they have no content at `new_sha`.
    An empty `changed_paths` short-circuits rather than running `git
    archive` with no pathspec, which would archive the entire tree.
    """
    if not changed_paths:
        return []
    result = subprocess.run(
        ["git", "archive", "--format=tar", new_sha, "--", *(p.as_posix() for p in changed_paths)],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        return [
            (PurePosixPath(member.name), archive.extractfile(member).read())
            for member in archive.getmembers()
            if member.isfile()
        ]
