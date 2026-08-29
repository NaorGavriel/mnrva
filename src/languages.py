from dataclasses import dataclass
from pathlib import PurePath
from prose_parser import _PROSE_FORMATS

@dataclass
class LanguageConfig:
    """Per-language tree-sitter node types used to route parsing and chunking."""

    language: str
    container_node_types: set[str]
    unit_node_types: set[str]
    import_node_types: set[str]


_PYTHON = LanguageConfig(
    language="python",
    container_node_types={"class_definition"},
    unit_node_types={"function_definition"},
    import_node_types={"import_statement", "import_from_statement"},
)

_TYPESCRIPT = LanguageConfig(
    language="typescript",
    container_node_types={"class_declaration", "interface_declaration"},
    # `arrow_function` catches arrow functions wherever they appear in the
    # tree (including nested in a `variable_declarator`), but the symbol
    # name for a named arrow function still has to be resolved from its
    # enclosing declarator, not from this node type alone. See
    # docs/ingest_index_pipeline.md §1.2 — under-chunked naming accepted for v1.
    unit_node_types={"function_declaration", "arrow_function"},
    import_node_types={"import_statement"},
)

_TSX = LanguageConfig(
    language="tsx",
    container_node_types={"class_declaration", "interface_declaration"},
    unit_node_types={"function_declaration", "arrow_function"},
    import_node_types={"import_statement"},
)

_JAVASCRIPT = LanguageConfig(
    language="javascript",
    container_node_types={"class_declaration"},
    unit_node_types={"function_declaration", "arrow_function"},
    import_node_types={"import_statement"},
)

LANGUAGE_CONFIG: dict[str, LanguageConfig] = {
    ".py": _PYTHON,
    ".ts": _TYPESCRIPT,
    ".tsx": _TSX,
    ".js": _JAVASCRIPT,
    ".jsx": _JAVASCRIPT,
}


def get_language(path: PurePath) -> str | None:
    """Return the canonical language name for `path`'s extension, or None if unsupported."""
    config = LANGUAGE_CONFIG.get(path.suffix)
    return config.language if config else None


def is_code_file(path: PurePath) -> bool:
    """Whether `path`'s extension is routed to tree-sitter chunking."""
    return path.suffix in LANGUAGE_CONFIG
