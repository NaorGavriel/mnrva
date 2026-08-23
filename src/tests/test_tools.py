import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

import chunks


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
