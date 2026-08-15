import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CHUNK_NAMESPACE = uuid.UUID("d6e5f8c2-4b1a-4e3a-9c7d-8a2b6f0e1d3c")

def make_chunk_id(path: Path, kind: str, class_name: str, symbol_name: str) -> str:
    """Derive a chunk's stable, content-independent id.

    `uuid5` over (path, kind, class_name, symbol_name) — deterministic, so
    editing a chunk's body never changes its id.
    """
    return str(
        uuid.uuid5(CHUNK_NAMESPACE, f"{path}{kind}{class_name}{symbol_name}")
    )


def make_content_hash(raw_text: str) -> str:
    """Fingerprint a chunk's body, used to decide whether refresh needs to re-embed."""
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

@dataclass
class Chunk:
    """A function/class/method-level unit of source code, plus its
    downstream enrichment and embedding once those stages have run."""

    id: str
    content_hash: str
    path: Path
    language: str
    kind: Literal["class", "function"]
    class_name: str
    symbol_name: str
    raw_text: str
    start_byte: int
    end_byte: int
    parent_id: str | None
    context_text: str | None = None
    embedding: list[float] | None = None


@dataclass
class ParsedFile:
    """The output of parsing one source file: its chunks plus the shared
    context (`source`, `imports`) enrichment needs but `Chunk` doesn't carry."""

    chunks: list[Chunk]
    source: str
    imports: list[str]
