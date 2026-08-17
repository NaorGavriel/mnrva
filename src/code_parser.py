from pathlib import Path, PurePath, PurePosixPath

from tree_sitter import Language, Node, Query, QueryCursor, Tree

from languages import LANGUAGE_CONFIG, LanguageConfig
from models import Chunk, ParsedFile, make_chunk
from registry import GrammarRegistry


def parse_code_file(
    read_from: Path, language: str, registry: GrammarRegistry, *, repo_root: Path | None = None
) -> ParsedFile:
    """Read, parse, and chunk a single source file.

    Thin file-level orchestrator: all extraction logic lives in the pure
    `extract_chunks`/`_extract_imports`.

    `repo_root` is stripped from `read_from` to produce the identity recorded on
    every chunk. keeps ids stable.
    """
    source = read_from.read_bytes()
    relative = read_from.relative_to(repo_root) if repo_root is not None else read_from
    path: PurePath = PurePosixPath(relative.as_posix())
    parser = registry.get_parser(language)
    tree = parser.parse(source)
    config = LANGUAGE_CONFIG[path.suffix]
    chunks = extract_chunks(tree, source, config, path)
    imports = _extract_imports(tree, source, config)
    return ParsedFile(chunks=chunks, source=source.decode("utf-8"), imports=imports)


def extract_chunks(
    tree: Tree, source: bytes, config: LanguageConfig, path: PurePath
) -> list[Chunk]:
    """Extract one chunk per container (class) and per unit (function/method).

    Two deliberately overlapping passes: containers first (so units
    can look up their enclosing container's chunk id), then units at any
    depth, including inside containers and inside other units.
    """
    root = tree.root_node
    language = tree.language
    chunks: list[Chunk] = []
    container_chunk_ids: dict[int, str] = {}

    for node in _query_nodes(language, root, config.container_node_types):
        class_name = _node_name(node, source)
        if not class_name:
            continue  # e.g. an anonymous default-exported class - no stable identity to chunk on
        chunk = make_chunk(
            path=path,
            language=config.language,
            kind="class",
            class_name=class_name,
            symbol_name="",
            raw_text=_node_text(node, source),
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            parent_id=None,
        )
        container_chunk_ids[node.id] = chunk.id
        chunks.append(chunk)

    for node in _query_nodes(language, root, config.unit_node_types):
        # A node with an unresolvable name (e.g. an inline/anonymous arrow-function callback)
        # is skipped because it causes chunk id collisions.
        symbol_name = _node_name(node, source)
        if not symbol_name:
            continue

        ancestor = _nearest_ancestor(node, config.container_node_types)
        if ancestor is not None:
            class_name = _node_name(ancestor, source)
            parent_id = container_chunk_ids.get(ancestor.id)
        else:
            class_name = ""
            parent_id = None

        chunk = make_chunk(
            path=path,
            language=config.language,
            kind="function",
            class_name=class_name,
            symbol_name=symbol_name,
            raw_text=_node_text(node, source),
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            parent_id=parent_id,
        )
        chunks.append(chunk)

    return chunks


_QUERY_CACHE: dict[tuple[Language, frozenset[str]], Query] = {}


def _get_query(language: Language, node_types: set[str]) -> Query:
    """Return the compiled `Query` matching any of `node_types`, building and
    caching it once per (language, node_types) pair so repeated calls don't
    pay recompilation cost."""
    key = (language, frozenset(node_types))
    query = _QUERY_CACHE.get(key)
    if query is None:
        query_source = "".join(f"({node_type}) @match\n" for node_type in node_types)
        query = Query(language, query_source)
        _QUERY_CACHE[key] = query
    return query


def _query_nodes(language: Language, root: Node, node_types: set[str]) -> list[Node]:
    """Collect every node under `root` (at any depth) whose type is in
    `node_types`, via tree-sitter's native Query engine."""
    query = _get_query(language, node_types)
    cursor = QueryCursor(query)
    return cursor.captures(root).get("match", [])


def _nearest_ancestor(node: Node, container_node_types: set[str]) -> Node | None:
    """Walk up from `node` to the nearest enclosing container node, or None if there isn't one."""
    current = node.parent
    while current is not None:
        if current.type in container_node_types:
            return current
        current = current.parent
    return None


def _node_name(node: Node, source: bytes) -> str:
    """Resolve `node`'s identifier, or "" if it has none.

    Falls back to the enclosing `variable_declarator`'s name for functions like
    `const handleClick = () => {...}`.
    """
    name_node = node.child_by_field_name("name")
    if name_node is None and node.type == "arrow_function" and node.parent is not None:
        if node.parent.type == "variable_declarator":
            name_node = node.parent.child_by_field_name("name")
    if name_node is None:
        return ""
    return _node_text(name_node, source)


def _node_text(node: Node, source: bytes) -> str:
    """Decode `node`'s own source span, verbatim."""
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _extract_imports(tree: Tree, source: bytes, config: LanguageConfig) -> list[str]:
    """Return the raw source text of every import statement in the file."""
    return [
        _node_text(node, source)
        for node in _query_nodes(tree.language, tree.root_node, config.import_node_types)
    ]
