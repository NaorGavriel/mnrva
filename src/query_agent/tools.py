import subprocess
from pathlib import Path
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from query_agent.agent_schemas import Citation, GrepMatch
from chunks import search_chunks
from models import ChunkSearchResult


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
) -> list[GrepMatch]:
    """Run `git grep -n --fixed-strings` for `pattern` inside the repo clone at `repo_path`.

    One mechanism for both identifier lookups and error-string lookups.
    Tool for "every place X is used" questions. Empty list for no matches.
    """
    command = ["git", "grep", "-n", "--fixed-strings", "-e", pattern]
    if file_glob is not None:
        command += ["--", file_glob]

    result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True)
    if result.returncode not in (0, 1):  # 1 = no matches, not an error
        raise RuntimeError(f"git grep failed: {result.stderr}")

    matches = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        file_path, line_number, line_text = parts
        matches.append(GrepMatch(file_path=file_path, line_number=int(line_number), line_text=line_text))
    return matches

def make_hybrid_search_tool(client: QdrantClient, collection_name: str) -> BaseTool:
    """Build the `hybrid_search_tool`, bound to a Qdrant `client`/`collection_name` (`chunks.search_chunks`)."""

    def hybrid_search(
        query: str,
        top_k: int = 10,
        language: str | None = None,
        kind: str | None = None,
    ) -> list[ChunkSearchResult]:
        """Hybrid dense + BM25 search over the indexed codebase. Returns the top matching chunks."""
        return search_chunks(client, collection_name, query, top_k, language=language, kind=kind)

    return tool("hybrid_search_tool")(hybrid_search)


def make_whole_file_read_tool(repo_path: Path) -> BaseTool:
    """Build the `whole_file_read_tool`, bound to the local repo clone at `repo_path`."""

    def read_file(
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read a file, or a 1-indexed inclusive line range of it, from the repository."""
        return whole_file_read_tool(repo_path, file_path, start_line, end_line)

    return tool("whole_file_read_tool")(read_file)


def make_grep_search_tool(repo_path: Path) -> BaseTool:
    """Build the `grep_search_tool`, bound to the local repo clone at `repo_path`."""

    def grep(pattern: str, file_glob: str | None = None) -> list[GrepMatch]:
        """Search the repository for a fixed-string pattern via `git grep`, optionally scoped to a glob."""
        return grep_search_tool(repo_path, pattern, file_glob)

    return tool("grep_search_tool")(grep)
