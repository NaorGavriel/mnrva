import configparser
import json
import re
import tomllib
from collections.abc import Callable
from pathlib import Path, PurePath, PurePosixPath
from typing import NamedTuple

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)

from models import ParsedFile, make_chunk

MAX_CHUNK_CHARS = 4000
CHUNK_OVERLAP = 200

_SECTION_HEADER = re.compile(r"^\s*\[+[^\[\]]+\]+\s*$")

_markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False
)
_json_splitter = RecursiveJsonSplitter(max_chunk_size=MAX_CHUNK_CHARS)
_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP
)


class _Section(NamedTuple):
    """One raw prose chunk before oversize-capping: name and text."""

    symbol_name: str
    text: str


def parse_prose_file(read_from: Path, *, repo_root: Path | None = None) -> ParsedFile:
    """Read and chunk a single non-code file, routed by extension to a format-aware splitter."""
    source = read_from.read_text(encoding="utf-8")
    relative = read_from.relative_to(repo_root) if repo_root is not None else read_from
    path: PurePath = PurePosixPath(relative.as_posix())

    language, split = _PROSE_FORMATS.get(path.suffix, ("text", _split_text))
    sections = _cap_oversized(split(source))
    chunks = [
        make_chunk(
            path=path,
            language=language,
            kind="section",
            class_name="",
            symbol_name=section.symbol_name,
            raw_text=section.text,
            start_byte=None,
            end_byte=None,
            start_line=None,
            end_line=None,
            parent_id=None,
        )
        for section in sections
    ]
    return ParsedFile(chunks=chunks, source=source, imports=[])


def _cap_oversized(sections: list[_Section]) -> list[_Section]:
    """Split any section still over `MAX_CHUNK_CHARS` into smaller parts."""
    capped: list[_Section] = []
    for section in sections:
        if len(section.text) <= MAX_CHUNK_CHARS:
            capped.append(section)
            continue
        parts = _fallback_splitter.split_text(section.text)
        capped.extend(
            _Section(f"{section.symbol_name} (part {i + 1})", part)
            for i, part in enumerate(parts)
        )
    return capped


def _split_markdown(text: str) -> list[_Section]:
    """Split markdown by header hierarchy (h1-h3), keeping headers in each chunk's text."""
    docs = _markdown_splitter.split_text(text)
    names = [" / ".join(doc.metadata.values()) or "document" for doc in docs]
    return [_Section(name, doc.page_content) for name, doc in zip(names, docs)]


def _split_json(text: str) -> list[_Section]:
    """Split JSON by top-level keys/array elements."""
    data = json.loads(text)
    pieces = _json_splitter.split_json(json_data=data, convert_lists=True)
    return [
        _Section(f"$[{i}]", json.dumps(piece, indent=2)) for i, piece in enumerate(pieces)
    ]


def _split_text(text: str) -> list[_Section]:
    """Split free-form text on paragraph/line boundaries; also the fallback for other extensions."""
    parts = _fallback_splitter.split_text(text)
    return [_Section(f"part-{i + 1}", part) for i, part in enumerate(parts)]


def _split_toml(text: str) -> list[_Section]:
    """Slice TOML into one chunk per table, cross-checked against tomllib's real structure."""
    data = tomllib.loads(text)  # raises tomllib.TOMLDecodeError on malformed input
    real_paths = _flatten_table_paths(data)
    sections = _split_bracketed_sections(text)
    for section in sections:
        # tomllib validates the file, not whether the regex below sliced it
        # correctly - cross-check each detected name against tomllib's real
        # dotted paths so a misread bracket (e.g. an array element alone on
        # its own line) fails loudly instead of silently mis-chunking.
        name = section.symbol_name.split("[", 1)[0]  # strip any _dedupe_names suffix
        if name not in ("preamble", "document") and name not in real_paths:
            raise ValueError(
                f"TOML section {section.symbol_name!r} doesn't match tomllib's real structure"
            )
    return sections


def _flatten_table_paths(data: dict, prefix: str = "") -> set[str]:
    """Every dotted key path reachable in a parsed TOML dict, at any depth."""
    paths: set[str] = set()
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        paths.add(path)
        if isinstance(value, dict):
            paths |= _flatten_table_paths(value, path)
    return paths


def _split_ini(text: str) -> list[_Section]:
    """Slice INI/CFG text into one chunk per `[section]`, validating each via `configparser`."""
    sections = _split_bracketed_sections(text)
    for section in sections:
        # Validated per-section rather than on the whole file: a leading
        # preamble of bare `key = value` lines is valid inside a section but
        # configparser rejects it as a standalone document.
        probe = (
            section.text
            if section.text.lstrip().startswith("[")
            else f"[{section.symbol_name}]\n{section.text}"
        )
        configparser.ConfigParser().read_string(probe)  # raises configparser.Error on malformed input
    return sections


def _split_bracketed_sections(text: str) -> list[_Section]:
    """Slice raw text at `[section]`/`[[table]]` header lines; shared by TOML and INI/CFG."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if _SECTION_HEADER.match(line)]

    if not starts:
        return [_Section("document", text)]

    sections: list[_Section] = []
    if starts[0] > 0:
        sections.append(_Section("preamble", "".join(lines[: starts[0]])))
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        name = lines[start].strip().strip("[]")
        sections.append(_Section(name, "".join(lines[start:end])))
    return _dedupe_names(sections)


def _dedupe_names(sections: list[_Section]) -> list[_Section]:
    """Disambiguate repeated section names with a `[0]`, `[1]`, ... suffix."""
    counts: dict[str, int] = {}
    for section in sections:
        counts[section.symbol_name] = counts.get(section.symbol_name, 0) + 1

    seen: dict[str, int] = {}
    deduped: list[_Section] = []
    for section in sections:
        if counts[section.symbol_name] == 1:
            deduped.append(section)
            continue
        index = seen.get(section.symbol_name, 0)
        seen[section.symbol_name] = index + 1
        deduped.append(section._replace(symbol_name=f"{section.symbol_name}[{index}]"))
    return deduped


_ProseFormat = tuple[str, Callable[[str], list[_Section]]]
"""A prose extension's (language name, splitter function) pair."""

_PROSE_FORMATS: dict[str, _ProseFormat] = {
    ".md": ("markdown", _split_markdown),
    ".json": ("json", _split_json),
    ".toml": ("toml", _split_toml),
    ".ini": ("ini", _split_ini),
    ".cfg": ("ini", _split_ini),
    ".txt": ("text", _split_text),
}

PROSE_EXTENSIONS = _PROSE_FORMATS.keys()

def is_prose_file(path: Path) -> bool:
    """Whether `path`'s extension is routed to prose/semantic chunking."""
    return path.suffix in _PROSE_FORMATS.keys()
