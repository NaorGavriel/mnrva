import subprocess
from pathlib import Path


def whole_file_read_tool(
    repo_path: Path,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read a file, or a 1-indexed inclusive line range of it, from the local repo clone at `repo_path`.

    Whole file if both `start_line` and `end_line` are omitted.
    """
    if start_line is not None and start_line < 1:
        raise ValueError(f"start_line must be >= 1, got {start_line}")
    if end_line is not None and end_line < 1:
        raise ValueError(f"end_line must be >= 1, got {end_line}")

    repo_root = repo_path.resolve()
    resolved = (repo_path / file_path).resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"{file_path!r} resolves outside the repository clone")
    if not resolved.is_file():
        raise FileNotFoundError(f"no such file: {file_path!r}")

    if start_line is None and end_line is None:
        return resolved.read_text(encoding="utf-8")

    lines = resolved.read_text(encoding="utf-8").splitlines(keepends=True)
    start = (start_line or 1) - 1
    end = end_line if end_line is not None else len(lines)
    return "".join(lines[start:end])


def grep_search_tool(
    repo_path: Path,
    pattern: str,
    file_glob: str | None = None,
) -> str:
    """Run `git grep -n --fixed-strings` for `pattern` inside the repo clone at `repo_path`.

    One mechanism for both identifier lookups and error-string lookups.
    Returns matching lines as `git grep` prints them; empty string for no matches.
    """
    command = ["git", "grep", "-n", "--fixed-strings", "-e", pattern]
    if file_glob is not None:
        command += ["--", file_glob]

    result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True)
    if result.returncode not in (0, 1):  # 1 = no matches, not an error
        raise RuntimeError(f"git grep failed: {result.stderr}")
    return result.stdout
