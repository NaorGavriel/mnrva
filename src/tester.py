from pathlib import Path

import db
from code_parser import extract_chunks, parse_code_file
from embeddings import embed_text
from enrichment import enrich_chunks
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
    parsed = parse_code_file(path, language, registry)

    
    print(f"Imports : \n{parsed.imports}")
    print(f"Source code : \n{parsed.source}")

    print(f"parsed {len(parsed.chunks)} chunks from {path}")
    for chunk in parsed.chunks:
        print(
            f"  kind={chunk.kind:9} class_name={chunk.class_name!r:12} "
            f"symbol_name={chunk.symbol_name!r:16} parent_id={chunk.parent_id}"
        )


def test_enrich_chunks() -> None:
    """Exercise enrich_chunks end to end against the live OpenAI API."""
    path = Path(__file__).parent / "models.py"
    language = get_language(path)
    assert language is not None, f"no language config for {path}"

    registry = LanguageRegistry()
    parsed = parse_code_file(path, language, registry)
    
    enriched = enrich_chunks(parsed.chunks, parsed.source, parsed.imports)
    print(f"enriched {len(enriched)} chunks from {path}")
    for chunk in enriched:
        assert chunk.context_text, "context_text was not populated"
        if chunk.kind == "class" :
            print(f"kind={chunk.kind:9} class_name={chunk.class_name!r:12}")
        else:
            print(f"kind={chunk.kind:9} symbol_name={chunk.symbol_name!r:16}")
        print(f"context_text={chunk.context_text!r}")


if __name__ == "__main__":
    #test_embedding()
    #test_db_init()
    #test_extract_chunks_inline()
    #test_parse_code_file()
    test_enrich_chunks()
