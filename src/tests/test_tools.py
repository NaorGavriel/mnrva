import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

import chunks
from query_agent.agent_schemas import GrepMatch
from query_agent.tools import (
    grep_search_tool,
    make_grep_search_tool,
    make_hybrid_search_tool,
    make_whole_file_read_tool,
    whole_file_read_tool,
)


class FakePoint:
    """Stands in for a Qdrant `ScoredPoint`."""

    def __init__(self, payload: dict, score: float, id: str = "11111111-1111-1111-1111-111111111111") -> None:
        self.payload = payload
        self.score = score
        self.id = id


class FakeQueryResponse:
    """Stands in for a Qdrant `QueryResponse`."""

    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class FakeQdrantClient:
    """Records query_points calls and returns one canned hit, instead of hitting a real Qdrant server."""

    def __init__(self) -> None:
        self.query_points_calls: list[dict] = []

    def query_points(self, **kwargs) -> FakeQueryResponse:
        self.query_points_calls.append(kwargs)
        return FakeQueryResponse(
            [
                FakePoint(
                    {
                        "file_path": "src/main.py",
                        "symbol_name": "greet",
                        "class_name": "",
                        "kind": "function",
                        "start_byte": 0,
                        "end_byte": 18,
                        "start_line": 1,
                        "end_line": 1,
                        "raw_text": "def greet(): pass",
                        "context_text": "greets someone",
                    },
                    score=0.9,
                )
            ]
        )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal real git repository on disk, so grep_search_tool can run real `git grep`."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    (repo_path / "main.py").write_text("def greet():\n    return 'hello'\n\n\ndef farewell():\n    return 'bye'\n")
    (repo_path / "nested").mkdir()
    (repo_path / "nested" / "utils.py").write_text("def greet_loudly():\n    return 'HELLO'\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-m", "initial"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    return repo_path


def test_whole_file_read_tool_returns_the_whole_file_when_no_lines_given(repo: Path) -> None:
    content = whole_file_read_tool(repo, "main.py")

    assert content == "def greet():\n    return 'hello'\n\n\ndef farewell():\n    return 'bye'\n"


def test_whole_file_read_tool_reads_a_nested_file(repo: Path) -> None:
    content = whole_file_read_tool(repo, "nested/utils.py")

    assert content == "def greet_loudly():\n    return 'HELLO'\n"


def test_whole_file_read_tool_slices_by_line_range_inclusive(repo: Path) -> None:
    content = whole_file_read_tool(repo, "main.py", start_line=1, end_line=2)

    assert content == "def greet():\n    return 'hello'\n"


def test_whole_file_read_tool_defaults_start_line_to_one(repo: Path) -> None:
    content = whole_file_read_tool(repo, "main.py", end_line=1)

    assert content == "def greet():\n"


def test_whole_file_read_tool_defaults_end_line_to_end_of_file(repo: Path) -> None:
    content = whole_file_read_tool(repo, "main.py", start_line=5)

    assert content == "def farewell():\n    return 'bye'\n"


def test_whole_file_read_tool_raises_for_a_missing_file(repo: Path) -> None:
    with pytest.raises(FileNotFoundError):
        whole_file_read_tool(repo, "does_not_exist.py")


def test_whole_file_read_tool_rejects_a_path_that_escapes_the_clone(repo: Path) -> None:
    with pytest.raises(ValueError):
        whole_file_read_tool(repo, "../outside.py")


def test_whole_file_read_tool_rejects_an_absolute_path_outside_the_clone(repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("secret\n")

    with pytest.raises(ValueError):
        whole_file_read_tool(repo, str(outside))


def test_whole_file_read_tool_rejects_a_start_line_below_one(repo: Path) -> None:
    with pytest.raises(ValueError):
        whole_file_read_tool(repo, "main.py", start_line=0)


def test_grep_search_tool_finds_matching_lines_with_line_numbers(repo: Path) -> None:
    matches = grep_search_tool(repo, "def greet")

    assert GrepMatch(file_path="main.py", line_number=1, line_text="def greet():") in matches
    assert GrepMatch(file_path="nested/utils.py", line_number=1, line_text="def greet_loudly():") in matches


def test_grep_search_tool_restricts_matches_to_the_given_glob(repo: Path) -> None:
    matches = grep_search_tool(repo, "def greet", file_glob="nested/*")

    assert any(match.file_path == "nested/utils.py" for match in matches)
    assert not any(match.file_path == "main.py" for match in matches)


def test_grep_search_tool_returns_empty_list_for_no_matches(repo: Path) -> None:
    matches = grep_search_tool(repo, "no_such_symbol_anywhere")

    assert matches == []


def test_make_hybrid_search_tool_has_the_documented_name() -> None:
    hybrid_search_tool = make_hybrid_search_tool(FakeQdrantClient(), "code_chunks")

    assert hybrid_search_tool.name == "hybrid_search_tool"


def test_make_hybrid_search_tool_delegates_to_search_chunks_with_the_bound_client_and_collection(monkeypatch) -> None:
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])
    client = FakeQdrantClient()
    hybrid_search_tool = make_hybrid_search_tool(client, "code_chunks")

    hybrid_search_tool.invoke({"query": "auth handling"})

    assert client.query_points_calls[0]["collection_name"] == "code_chunks"


def test_make_hybrid_search_tool_returns_search_chunks_output_unmodified(monkeypatch) -> None:
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])
    hybrid_search_tool = make_hybrid_search_tool(FakeQdrantClient(), "code_chunks")

    result = hybrid_search_tool.invoke({"query": "auth handling"})
    assert result == [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "file_path": "src/main.py",
            "symbol_name": "greet",
            "class_name": "",
            "kind": "function",
            "start_byte": 0,
            "end_byte": 18,
            "start_line": 1,
            "end_line": 1,
            "raw_text": "def greet(): pass",
            "context_text": "greets someone",
            "score": 0.9,
        }
    ]


def test_make_whole_file_read_tool_has_the_documented_name(repo: Path) -> None:
    read_tool = make_whole_file_read_tool(repo)

    assert read_tool.name == "whole_file_read_tool"


def test_make_whole_file_read_tool_delegates_to_the_bound_repo_path(repo: Path) -> None:
    read_tool = make_whole_file_read_tool(repo)

    content = read_tool.invoke({"file_path": "main.py"})

    assert content == "def greet():\n    return 'hello'\n\n\ndef farewell():\n    return 'bye'\n"


def test_make_grep_search_tool_has_the_documented_name(repo: Path) -> None:
    grep_tool = make_grep_search_tool(repo)

    assert grep_tool.name == "grep_search_tool"


def test_make_grep_search_tool_delegates_to_the_bound_repo_path(repo: Path) -> None:
    grep_tool = make_grep_search_tool(repo)

    matches = grep_tool.invoke({"pattern": "def greet"})

    assert GrepMatch(file_path="main.py", line_number=1, line_text="def greet():") in matches
