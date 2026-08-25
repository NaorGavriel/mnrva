import hashlib
import uuid
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal, TypedDict


CHUNK_NAMESPACE = uuid.UUID("d6e5f8c2-4b1a-4e3a-9c7d-8a2b6f0e1d3c")

def make_chunk_id(path: PurePath, kind: str, class_name: str, symbol_name: str) -> str:
    """Derive a chunk's stable, content-independent id.

    `uuid5` over (path, kind, class_name, symbol_name) - deterministic, so
    editing a chunk's body never changes its id.
    """
    return str(
        uuid.uuid5(
            CHUNK_NAMESPACE, f"{path.as_posix()}{kind}{class_name}{symbol_name}"
        )
    )


def make_content_hash(raw_text: str) -> str:
    """Fingerprint a chunk's body, used to decide whether refresh needs to re-embed."""
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def chunk_retrieval_text(chunk: "Chunk") -> str:
    """The text that represents `chunk` for retrieval: its enrichment context
    prepended to its raw body.

    Shared by dense embedding and BM25 indexing so both halves of hybrid
    search see identical content for the same chunk.
    """
    if chunk.context_text:
        return f"{chunk.context_text}\n\n{chunk.raw_text}"
    return chunk.raw_text

@dataclass
class Chunk:
    """A function/class/method-level unit of source code, or a section of a
    prose file, plus its downstream enrichment and embedding once those
    stages have run."""

    id: str
    content_hash: str
    path: PurePath
    language: str
    kind: Literal["class", "function", "section"]
    class_name: str
    symbol_name: str
    raw_text: str
    start_byte: int | None
    end_byte: int | None
    start_line: int | None
    end_line: int | None
    parent_id: str | None
    context_text: str | None = None
    embedding: list[float] | None = None


def make_chunk(
    *,
    path: PurePath,
    language: str,
    kind: Literal["class", "function", "section"],
    class_name: str,
    symbol_name: str,
    raw_text: str,
    start_byte: int | None,
    end_byte: int | None,
    start_line: int | None,
    end_line: int | None,
    parent_id: str | None,
) -> Chunk:
    """Build a `Chunk`, deriving its id and content hash from the given identity and body.

    The single construction point for `Chunk`, shared by `code_parser.py` and `prose_parser.py`.
    """
    return Chunk(
        id=make_chunk_id(path, kind, class_name, symbol_name),
        content_hash=make_content_hash(raw_text),
        path=path,
        language=language,
        kind=kind,
        class_name=class_name,
        symbol_name=symbol_name,
        raw_text=raw_text,
        start_byte=start_byte,
        end_byte=end_byte,
        start_line=start_line,
        end_line=end_line,
        parent_id=parent_id,
    )


class ChunkSearchResult(TypedDict):
    """One hybrid-search hit: a chunk's identifying/text payload plus its fused RRF score."""

    id: str
    file_path: str
    symbol_name: str
    class_name: str
    kind: Literal["class", "function", "section"]
    start_byte: int | None
    end_byte: int | None
    start_line: int | None
    end_line: int | None
    raw_text: str
    context_text: str | None
    score: float


@dataclass
class ParsedFile:
    """The output of parsing one source file: its chunks plus the shared
    context (`source`, `imports`) enrichment needs but `Chunk` doesn't carry.

    `path` is kept even though every chunk also carries it, so a file's
    identity survives when it parses to zero chunks (e.g. an empty file).
    """

    path: PurePath
    chunks: list[Chunk]
    source: str
    imports: list[str]
