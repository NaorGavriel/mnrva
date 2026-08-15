from pathlib import Path

import db
from code_parser import extract_chunks, parse_code_file
from embeddings import embed_text
from languages import LANGUAGE_CONFIG, get_language
from registry import LanguageRegistry


def test_embedding() -> None:
    """Smoke-test embeddings.embed_text against the live OpenAI API."""
    vector = embed_text("hello world")
    print(f"embedding length: {len(vector)}")
    print(f"first 5 values: {vector[:5]}")


def test_db_init() -> None:
    """Smoke-test db.init_client/ensure_collection against a live Qdrant instance."""
    client = db.init_client(url=db.QDRANT_URL)
    db.ensure_collection(client, db.COLLECTION_NAME)
    print(f"collection '{db.COLLECTION_NAME}' ready at {db.QDRANT_URL}")


def test_extract_chunks_inline() -> None:
    """Exercise extract_chunks directly on inline source — no disk I/O."""
    source = (
        "class Greeter:\n"
        "    def greet(self, name):\n"
        "        return f'hello {name}'\n"
        "\n"
        "def standalone():\n"
        "    pass\n"
    ).encode("utf-8")

    registry = LanguageRegistry()
    parser = registry.get_parser("python")
    tree = parser.parse(source)
    config = LANGUAGE_CONFIG[".py"]
    chunks = extract_chunks(tree, source, config, Path("inline_test.py"))

    print(f"extracted {len(chunks)} chunks from inline source")
    for chunk in chunks:
        assert chunk.raw_text == source[chunk.start_byte : chunk.end_byte].decode(
            "utf-8"
        ), "raw_text invariant broken"
        print(
            f"  kind={chunk.kind:9} class_name={chunk.class_name!r:10} "
            f"symbol_name={chunk.symbol_name!r:12} parent_id={chunk.parent_id}"
        )


def test_parse_code_file() -> None:
    """Exercise parse_code_file end to end against a real file on disk."""
    path = Path(__file__).parent / "models.py"
    language = get_language(path)
    assert language is not None, f"no language config for {path}"

    registry = LanguageRegistry()
    chunks = parse_code_file(path, language, registry)

    print(f"parsed {len(chunks)} chunks from {path}")
    for chunk in chunks:
        print(
            f"  kind={chunk.kind:9} class_name={chunk.class_name!r:12} "
            f"symbol_name={chunk.symbol_name!r:16} parent_id={chunk.parent_id}"
        )


if __name__ == "__main__":
    test_embedding()
    test_db_init()
    test_extract_chunks_inline()
    test_parse_code_file()
