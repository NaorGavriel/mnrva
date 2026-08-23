from query_agent.tools import make_get_chunks_by_id_tool


class FakeRecord:
    """Stands in for a Qdrant `Record`, as returned by `retrieve`."""

    def __init__(self, payload: dict, id: str) -> None:
        self.payload = payload
        self.id = id


class FakeQdrantClient:
    """Records `retrieve` calls and returns a canned response, instead of hitting a real Qdrant server."""

    def __init__(self, retrieve_response: list[FakeRecord] | None = None) -> None:
        self.retrieve_calls: list[dict] = []
        self._retrieve_response = retrieve_response if retrieve_response is not None else []

    def retrieve(self, **kwargs) -> list[FakeRecord]:
        self.retrieve_calls.append(kwargs)
        return self._retrieve_response


def _make_record(chunk_id: str, **payload_overrides) -> FakeRecord:
    payload = dict(
        file_path="src/main.py",
        symbol_name="greet",
        class_name="",
        kind="function",
        start_byte=0,
        end_byte=18,
        start_line=1,
        end_line=1,
        raw_text="def greet(): pass",
        context_text="greets someone",
        language="python",
        parent_id=None,
        content_hash="hash",
    )
    payload.update(payload_overrides)
    return FakeRecord(payload, id=chunk_id)


def test_get_chunks_by_id_tool_calls_retrieve_with_the_bound_collection_name() -> None:
    client = FakeQdrantClient()
    tool = make_get_chunks_by_id_tool(client, "code_chunks")

    tool.invoke({"chunk_ids": ["11111111-1111-1111-1111-111111111111"]})

    assert client.retrieve_calls[0]["collection_name"] == "code_chunks"


def test_get_chunks_by_id_tool_includes_found_chunk_content() -> None:
    client = FakeQdrantClient(
        retrieve_response=[_make_record("11111111-1111-1111-1111-111111111111", raw_text="def greet(): pass")]
    )
    tool = make_get_chunks_by_id_tool(client, "code_chunks")

    result = tool.invoke({"chunk_ids": ["11111111-1111-1111-1111-111111111111"]})

    assert "src/main.py" in result
    assert "def greet(): pass" in result


def test_get_chunks_by_id_tool_reports_ids_not_found_instead_of_raising() -> None:
    client = FakeQdrantClient(retrieve_response=[])

    tool = make_get_chunks_by_id_tool(client, "code_chunks")
    result = tool.invoke({"chunk_ids": ["99999999-9999-9999-9999-999999999999"]})

    assert "Not found" in result
    assert "99999999-9999-9999-9999-999999999999" in result


def test_get_chunks_by_id_tool_reports_only_the_missing_ids_when_some_are_found() -> None:
    client = FakeQdrantClient(
        retrieve_response=[_make_record("11111111-1111-1111-1111-111111111111")]
    )
    tool = make_get_chunks_by_id_tool(client, "code_chunks")

    result = tool.invoke(
        {"chunk_ids": ["11111111-1111-1111-1111-111111111111", "99999999-9999-9999-9999-999999999999"]}
    )

    assert "99999999-9999-9999-9999-999999999999" in result
    assert "src/main.py" in result
