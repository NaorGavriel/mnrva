import subprocess
from pathlib import Path

import pytest

from query_agent.tools import grep_search_tool, whole_file_read_tool


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
    output = grep_search_tool(repo, "def greet")

    assert "main.py:1:def greet():" in output
    assert "nested/utils.py:1:def greet_loudly():" in output


def test_grep_search_tool_treats_the_pattern_as_a_fixed_string_not_a_regex(repo: Path) -> None:
    output = grep_search_tool(repo, "greet()")

    assert "main.py:1:def greet():" in output


def test_grep_search_tool_restricts_matches_to_the_given_glob(repo: Path) -> None:
    output = grep_search_tool(repo, "def greet", file_glob="nested/*")

    assert "nested/utils.py" in output
    assert "main.py:1" not in output


def test_grep_search_tool_returns_empty_string_for_no_matches(repo: Path) -> None:
    output = grep_search_tool(repo, "no_such_symbol_anywhere")

    assert output == ""
