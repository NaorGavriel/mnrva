import subprocess
from pathlib import Path, PurePosixPath

import pytest

from registry import LanguageRegistry
from repository_ingester import parse_repository_files


@pytest.fixture
def repo_with_files(tmp_path: Path) -> Path:
    """A real git repo with a code file, a prose file, and an unwanted file."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo_path, check=True)

    (repo_path / "main.py").write_text("def greet():\n    return 'hi'\n")
    (repo_path / "README.md").write_text("# demo\n\nsome docs\n")
    (repo_path / "package-lock.json").write_text("{}\n")

    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    return repo_path


def test_parse_repository_files_parses_every_wanted_file(repo_with_files: Path) -> None:
    """One ParsedFile comes back per wanted file, routed to the right parser."""
    parsed_files = parse_repository_files(repo_with_files, LanguageRegistry())

    parsed_by_path = {parsed.path: parsed for parsed in parsed_files}
    assert set(parsed_by_path) == {PurePosixPath("main.py"), PurePosixPath("README.md")}

    code_parsed = parsed_by_path[PurePosixPath("main.py")]
    assert [chunk.symbol_name for chunk in code_parsed.chunks] == ["greet"]
    assert code_parsed.chunks[0].kind == "function"

    prose_parsed = parsed_by_path[PurePosixPath("README.md")]
    assert prose_parsed.chunks
    assert all(chunk.kind == "section" for chunk in prose_parsed.chunks)


def test_parse_repository_files_excludes_unwanted_files(repo_with_files: Path) -> None:
    """A denylisted file (e.g. a lockfile) never becomes a ParsedFile."""
    parsed_files = parse_repository_files(repo_with_files, LanguageRegistry())

    paths = {parsed.path for parsed in parsed_files}
    assert PurePosixPath("package-lock.json") not in paths