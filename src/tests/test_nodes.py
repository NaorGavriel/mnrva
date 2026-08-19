from langchain_core.messages import AIMessage

import chunks
from query_agent.nodes import make_generate_answer_node, make_retrieve_documents_node
from query_agent.state import AgentState


class FakeRunnable:
    """A minimal stand-in for a LangChain Runnable: records the input, returns a canned output."""

    def __init__(self, output) -> None:
        self.output = output
        self.invoke_calls: list = []

    def invoke(self, input, *args, **kwargs):
        self.invoke_calls.append(input)
        return self.output


class FakePoint:
    """Stands in for a Qdrant `ScoredPoint`."""

    def __init__(self, payload: dict, score: float) -> None:
        self.payload = payload
        self.score = score


class FakeQueryResponse:
    """Stands in for a Qdrant `QueryResponse`."""

    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class FakeQdrantClient:
    """Records query_points calls and returns a canned response, instead of hitting a real Qdrant server."""

    def __init__(self, query_response: FakeQueryResponse | None = None) -> None:
        self.query_points_calls: list[dict] = []
        self._query_response = query_response if query_response is not None else FakeQueryResponse([])

    def query_points(self, **kwargs) -> FakeQueryResponse:
        self.query_points_calls.append(kwargs)
        return self._query_response


def _make_point(score: float = 0.75, **payload_overrides) -> FakePoint:
    payload = dict(
        file_path="src/auth.py",
        symbol_name="authenticate",
        class_name="",
        kind="function",
        start_byte=0,
        end_byte=18,
        raw_text="def authenticate(): ...",
        context_text=None,
    )
    payload.update(payload_overrides)
    return FakePoint(payload, score)


def test_retrieve_documents_node_searches_with_the_question_and_returns_chunks(monkeypatch) -> None:
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])
    client = FakeQdrantClient(query_response=FakeQueryResponse([_make_point(score=0.9)]))
    node = make_retrieve_documents_node(client, "code_chunks")
    state: AgentState = {"question": "how does auth work?", "retrieved_chunks": [], "answer": ""}

    result = node(state)

    assert len(result["retrieved_chunks"]) == 1
    chunk = result["retrieved_chunks"][0]
    assert chunk["file_path"] == "src/auth.py"
    assert chunk["symbol_name"] == "authenticate"
    assert chunk["score"] == 0.9


def test_generate_answer_node_builds_a_prompt_from_the_question_and_chunks() -> None:
    llm = FakeRunnable(AIMessage(content="Auth is handled in authenticate()."))
    node = make_generate_answer_node(llm)
    state: AgentState = {
        "question": "how does auth work?",
        "retrieved_chunks": [
            {
                "file_path": "src/auth.py",
                "symbol_name": "authenticate",
                "class_name": "",
                "kind": "function",
                "start_byte": 0,
                "end_byte": 18,
                "raw_text": "def authenticate(): ...",
                "context_text": None,
                "score": 0.9,
            }
        ],
        "answer": "",
    }

    result = node(state)

    assert result == {"answer": "Auth is handled in authenticate()."}
    prompt = llm.invoke_calls[0]
    assert "how does auth work?" in prompt
    assert "src/auth.py" in prompt
    assert "def authenticate(): ..." in prompt
