import json
import tomllib
from pathlib import Path, PurePosixPath

import pytest

from models import ParsedFile, make_chunk_id
from prose_parser import MAX_CHUNK_CHARS, _split_toml, parse_prose_file

EXAMPLES_DIR = Path(__file__).parent / "prose_parse_examples"


def test_split_toml_one_chunk_per_table() -> None:
    """Each [table] in the source becomes its own section, named by its dotted path."""
    text = """\
[project]
name = "demo"

[project.scripts]
demo = "demo:main"
"""

    sections = _split_toml(text)

    assert [section.symbol_name for section in sections] == ["project", "project.scripts"]


def test_split_toml_raises_on_bracket_false_positive() -> None:
    """A bracket-wrapped array element alone on its own line is valid TOML but not
    a real header - the tomllib cross-check must catch the regex misreading it as
    one, instead of silently mis-chunking."""
    text = """\
[tool.some_plugin]
grid = [
    ["a", "b"],
    ["c", "d"]
]
"""

    with pytest.raises(ValueError):
        _split_toml(text)


def test_split_toml_raises_on_malformed_syntax() -> None:
    """Invalid TOML fails loudly via tomllib, before the regex scan even runs."""
    with pytest.raises(tomllib.TOMLDecodeError):
        _split_toml("not = valid = toml")


def test_split_toml_dedupes_array_of_tables() -> None:
    """Repeated [[array_of_tables]] headers get distinct symbol_names (see _dedupe_names)."""
    text = """\
[[tool.some_plugin.rules]]
name = "a"

[[tool.some_plugin.rules]]
name = "b"
"""

    names = [section.symbol_name for section in _split_toml(text)]

    assert names == ["tool.some_plugin.rules[0]", "tool.some_plugin.rules[1]"]


def test_split_toml_dedup_prevents_chunk_id_collision() -> None:
    """Deduped symbol_names produce distinct chunk ids end-to-end, so repeated
    [[array_of_tables]] entries don't overwrite each other in Qdrant."""
    text = """\
[[tool.some_plugin.rules]]
name = "a"

[[tool.some_plugin.rules]]
name = "b"
"""
    path = PurePosixPath("pyproject.toml")

    sections = _split_toml(text)
    ids = {make_chunk_id(path, "section", "", section.symbol_name) for section in sections}

    assert len(ids) == len(sections)


def _parse_example(filename: str) -> ParsedFile:
    """Run parse_prose_file end to end against a fixture under EXAMPLES_DIR."""
    return parse_prose_file(EXAMPLES_DIR / filename)


def _assert_common_invariants(parsed: ParsedFile, language: str) -> None:
    """Properties every parse_prose_file result must hold, regardless of format."""
    assert parsed.chunks, "expected at least one chunk"

    ids = {chunk.id for chunk in parsed.chunks}
    assert len(ids) == len(parsed.chunks), "duplicate chunk ids within one file"

    for chunk in parsed.chunks:
        assert chunk.kind == "section"
        assert chunk.language == language
        assert chunk.symbol_name
        assert chunk.raw_text, "empty chunk"
        assert len(chunk.raw_text) <= MAX_CHUNK_CHARS, "_cap_oversized bound violated"


def test_split_markdown() -> None:
    """example.md splits by header hierarchy, keeping header lines in the chunk text."""
    parsed = _parse_example("example.md")
    _assert_common_invariants(parsed, "markdown")

    assert len(parsed.chunks) > 1
    joined = "\n".join(chunk.raw_text for chunk in parsed.chunks)
    assert "## 2. System Components" in joined
    assert "### 2.1 Ingestion & Indexing Pipeline" in joined
    assert any(
        "2.1 Ingestion & Indexing Pipeline" in chunk.symbol_name for chunk in parsed.chunks
    )


def test_split_toml_example_file() -> None:
    """example.toml gets one chunk per [table]/[[array-of-tables]] entry."""
    parsed = _parse_example("example.toml")
    _assert_common_invariants(parsed, "toml")

    names = [chunk.symbol_name for chunk in parsed.chunks]
    assert names == [
        "project",
        "project.scripts",
        "build-system",
        "tool.pytest.ini_options",
        "tool.some_plugin",
        "tool.some_plugin.rules[0]",
        "tool.some_plugin.rules[1]",
    ]


def test_split_ini() -> None:
    """example.ini's leading key/value pair (before any [section]) becomes a preamble chunk."""
    parsed = _parse_example("example.ini")
    _assert_common_invariants(parsed, "ini")

    names = [chunk.symbol_name for chunk in parsed.chunks]
    assert names == ["preamble", "database", "logging", "feature_flags"]


def test_split_cfg() -> None:
    """example.cfg gets one chunk per [section], including a dotted section name."""
    parsed = _parse_example("example.cfg")
    _assert_common_invariants(parsed, "ini")

    names = [chunk.symbol_name for chunk in parsed.chunks]
    assert names == ["metadata", "options", "options.extras_require"]


def test_split_txt() -> None:
    """example.txt is long enough to force the fallback splitter into multiple parts."""
    parsed = _parse_example("example.txt")
    _assert_common_invariants(parsed, "text")

    names = [chunk.symbol_name for chunk in parsed.chunks]
    assert len(names) > 1
    assert names == [f"part-{i + 1}" for i in range(len(names))]


def test_split_json() -> None:
    """example.json's chunks are each valid, re-serialized JSON covering every top-level key."""
    parsed = _parse_example("example.json")
    _assert_common_invariants(parsed, "json")

    original_keys = set(json.loads((EXAMPLES_DIR / "example.json").read_text()))
    found_keys: set[str] = set()
    for chunk in parsed.chunks:
        piece = json.loads(chunk.raw_text)
        assert isinstance(piece, dict), "top-level JSON pieces should stay dict-shaped"
        found_keys |= piece.keys()
    assert found_keys == original_keys
