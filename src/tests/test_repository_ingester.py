import subprocess
from pathlib import Path, PurePosixPath

import pytest

import repository_ingester
from models import Chunk, ParsedFile
from registry import LanguageRegistry
from repository_ingester import _enrich_embed_and_upsert, parse_repository_files


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


def _make_chunk(path: str, index: int, content_hash: str = "hash") -> Chunk:
    return Chunk(
        id=f"{path}#{index}",
        content_hash=content_hash,
        path=PurePosixPath(path),
        language="python",
        kind="function",
        class_name="",
        symbol_name=f"fn{index}",
        raw_text=f"def fn{index}(): pass",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=1,
        parent_id=None,
    )


def _make_parsed_file(path: str, n_chunks: int) -> ParsedFile:
    return ParsedFile(
        path=PurePosixPath(path),
        chunks=[_make_chunk(path, i) for i in range(n_chunks)],
        source="source",
        imports=[],
    )


async def _identity_enrich_chunks(chunks: list[Chunk], source: str, imports: list[str]) -> list[Chunk]:
    return chunks


async def _identity_embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return chunks


def _patch_pipeline_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embedding_batch_size: int,
    max_concurrency: int,
    existing_chunks: dict[str, Chunk] | None = None,
) -> list[list[str]]:
    """Wire up the pipeline's dependencies with in-memory fakes, returning the
    list of upserted batches (each a list of chunk ids, in upsert order)."""
    monkeypatch.setattr(repository_ingester, "EMBEDDING_BATCH_SIZE", embedding_batch_size)
    monkeypatch.setattr(repository_ingester, "ENRICHMENT_MAX_CONCURRENCY", max_concurrency)
    monkeypatch.setattr(repository_ingester, "SKIP_CHECK_BATCH_SIZE", 1000)
    monkeypatch.setattr(repository_ingester, "enrich_chunks", _identity_enrich_chunks)
    monkeypatch.setattr(repository_ingester, "embed_chunks", _identity_embed_chunks)

    existing_chunks = existing_chunks or {}
    monkeypatch.setattr(
        repository_ingester,
        "get_chunks_by_id",
        lambda client, collection_name, ids: [existing_chunks[id] for id in ids if id in existing_chunks],
    )

    upsert_batches: list[list[str]] = []
    monkeypatch.setattr(
        repository_ingester,
        "upsert_chunks",
        lambda client, collection_name, chunks: upsert_batches.append([chunk.id for chunk in chunks]),
    )
    return upsert_batches


def test_fetch_existing_content_hashes_batches_id_lookups_at_the_configured_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More chunk ids than SKIP_CHECK_BATCH_SIZE are looked up across multiple
    get_chunks_by_id calls, merged into one hash map - not one call per file,
    and not one call for the whole repo."""
    monkeypatch.setattr(repository_ingester, "SKIP_CHECK_BATCH_SIZE", 2)
    parsed_files = [_make_parsed_file("a.py", 2), _make_parsed_file("b.py", 1)]
    existing_chunks = {chunk.id: chunk for parsed in parsed_files for chunk in parsed.chunks}
    batch_sizes: list[int] = []

    def fake_get_chunks_by_id(client, collection_name, ids):
        batch_sizes.append(len(ids))
        return [existing_chunks[id] for id in ids if id in existing_chunks]

    monkeypatch.setattr(repository_ingester, "get_chunks_by_id", fake_get_chunks_by_id)

    existing_hashes = repository_ingester._fetch_existing_content_hashes(client=None, parsed_files=parsed_files)

    assert batch_sizes == [2, 1]
    assert existing_hashes == {chunk_id: chunk.content_hash for chunk_id, chunk in existing_chunks.items()}


async def test_pipeline_spans_embedding_batches_across_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single embedding batch can contain chunks from more than one file,
    packed to EMBEDDING_BATCH_SIZE rather than flushed per file."""
    upsert_batches = _patch_pipeline_dependencies(monkeypatch, embedding_batch_size=3, max_concurrency=1)
    parsed_files = [_make_parsed_file("a.py", 2), _make_parsed_file("b.py", 4)]

    total = await _enrich_embed_and_upsert(client=None, parsed_files=parsed_files)

    assert total == 6
    assert [len(batch) for batch in upsert_batches] == [3, 3]
    assert upsert_batches[0] == ["a.py#0", "a.py#1", "b.py#0"]
    assert upsert_batches[1] == ["b.py#1", "b.py#2", "b.py#3"]


async def test_pipeline_skips_a_file_with_no_chunks_without_enqueueing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file that parses to zero chunks (e.g. import-only) is skipped outright -
    there's nothing to check against Qdrant or to enrich."""
    empty_file = ParsedFile(path=PurePosixPath("empty.py"), chunks=[], source="import os", imports=["os"])
    has_work = _make_parsed_file("b.py", 1)
    upsert_batches = _patch_pipeline_dependencies(monkeypatch, embedding_batch_size=10, max_concurrency=1)

    total = await _enrich_embed_and_upsert(client=None, parsed_files=[empty_file, has_work])

    assert total == 1
    assert upsert_batches == [["b.py#0"]]


async def test_pipeline_flushes_a_partial_final_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunks left over after the last file, short of a full EMBEDDING_BATCH_SIZE,
    still get embedded and upserted as one final smaller batch."""
    upsert_batches = _patch_pipeline_dependencies(monkeypatch, embedding_batch_size=10, max_concurrency=1)
    parsed_files = [_make_parsed_file("a.py", 2), _make_parsed_file("b.py", 1)]

    total = await _enrich_embed_and_upsert(client=None, parsed_files=parsed_files)

    assert total == 3
    assert [len(batch) for batch in upsert_batches] == [3]


async def test_pipeline_skips_a_file_already_upserted_with_matching_content_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file whose chunks are all already in Qdrant with a matching content_hash
    is never enqueued for enrichment - the crash-recovery skip-check."""
    already_done = _make_parsed_file("done.py", 2)
    needs_work = _make_parsed_file("todo.py", 1)
    existing_chunks = {chunk.id: chunk for chunk in already_done.chunks}

    upsert_batches = _patch_pipeline_dependencies(
        monkeypatch, embedding_batch_size=10, max_concurrency=2, existing_chunks=existing_chunks
    )

    total = await _enrich_embed_and_upsert(client=None, parsed_files=[already_done, needs_work])

    assert total == 1
    assert upsert_batches == [["todo.py#0"]]


async def test_pipeline_reenriches_a_file_whose_content_hash_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk id that exists in Qdrant but with a stale content_hash still
    needs enrichment - the skip-check only skips exact matches."""
    changed = _make_parsed_file("changed.py", 1)
    stale_chunk = _make_chunk("changed.py", 0, content_hash="stale-hash")
    existing_chunks = {stale_chunk.id: stale_chunk}

    upsert_batches = _patch_pipeline_dependencies(
        monkeypatch, embedding_batch_size=10, max_concurrency=1, existing_chunks=existing_chunks
    )

    total = await _enrich_embed_and_upsert(client=None, parsed_files=[changed])

    assert total == 1
    assert upsert_batches == [["changed.py#0"]]