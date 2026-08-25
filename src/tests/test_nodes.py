from langchain_core.messages import AIMessage, HumanMessage

import chunks
from query_agent.schemas import (
    Answer,
    Citation,
    EvaluateAnswer,
    EvaluateQuestion,
    GeneratedAnswer,
    GradeDocument,
    QuestionFilters,
)
from query_agent.nodes import (
    make_build_conversation_window_node,
    make_evaluate_answer_node,
    make_evaluate_question_node,
    make_generate_answer_node,
    make_grade_documents_node,
    make_retrieve_documents_node,
)
from query_agent.state import AgentState


class FakeStructuredLLM:
    """Stands in for `llm.with_structured_output(...)`: returns canned outputs in call order, records inputs."""

    def __init__(self, outputs: list) -> None:
        self._outputs = list(outputs)
        self.invoke_calls: list = []

    def invoke(self, input, *args, **kwargs):
        self.invoke_calls.append(input)
        return self._outputs.pop(0)

    def batch(self, inputs, *args, **kwargs):
        self.invoke_calls.extend(inputs)
        outputs, self._outputs = self._outputs[: len(inputs)], self._outputs[len(inputs) :]
        return outputs


class FakeLLM:
    """Stands in for a chat-model Runnable whose only use in these nodes is with_structured_output."""

    def __init__(self, outputs: list) -> None:
        self.structured = FakeStructuredLLM(outputs)

    def with_structured_output(self, schema):
        return self.structured


class FakePoint:
    """Stands in for a Qdrant `ScoredPoint`."""

    def __init__(self, payload: dict, score: float, id: str = "11111111-1111-1111-1111-111111111111") -> None:
        self.payload = payload
        self.score = score
        self.id = id


class FakeQueryResponse:
    """Stands in for a Qdrant `QueryResponse`."""

    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class FakeRecord:
    """Stands in for a Qdrant `Record`, as returned by `retrieve`."""

    def __init__(self, payload: dict, id: str) -> None:
        self.payload = payload
        self.id = id


class FakeQdrantClient:
    """Records query_points/retrieve calls and returns canned responses, instead of hitting a real Qdrant server."""

    def __init__(
        self,
        query_response: FakeQueryResponse | None = None,
        retrieve_response: list[FakeRecord] | None = None,
    ) -> None:
        self.query_points_calls: list[dict] = []
        self.retrieve_calls: list[dict] = []
        self._query_response = query_response if query_response is not None else FakeQueryResponse([])
        self._retrieve_response = retrieve_response if retrieve_response is not None else []

    def query_points(self, **kwargs) -> FakeQueryResponse:
        self.query_points_calls.append(kwargs)
        return self._query_response

    def retrieve(self, **kwargs) -> list[FakeRecord]:
        self.retrieve_calls.append(kwargs)
        return self._retrieve_response


def _make_point(score: float = 0.75, **payload_overrides) -> FakePoint:
    payload = dict(
        file_path="src/auth.py",
        symbol_name="authenticate",
        class_name="",
        kind="function",
        start_byte=0,
        end_byte=18,
        start_line=1,
        end_line=1,
        raw_text="def authenticate(): ...",
        context_text=None,
    )
    payload.update(payload_overrides)
    return FakePoint(payload, score)


def _make_record(chunk_id: str, **payload_overrides) -> FakeRecord:
    payload = dict(
        file_path="src/auth.py",
        symbol_name="authenticate",
        class_name="",
        kind="function",
        start_byte=0,
        end_byte=18,
        start_line=1,
        end_line=1,
        raw_text="def authenticate(): ...",
        context_text="authentication function",
        language="python",
        parent_id=None,
        content_hash="hash",
    )
    payload.update(payload_overrides)
    return FakeRecord(payload, id=chunk_id)


def _search_result(*, id: str = "id-1", file_path: str = "src/auth.py", symbol_name: str = "authenticate", raw_text: str = "def authenticate(): ...") -> dict:
    return {
        "id": id,
        "file_path": file_path,
        "symbol_name": symbol_name,
        "class_name": "",
        "kind": "function",
        "start_byte": 0,
        "end_byte": 18,
        "start_line": 1,
        "end_line": 1,
        "raw_text": raw_text,
        "context_text": None,
        "score": 0.9,
    }


def test_build_conversation_window_node_is_noop_on_the_first_turn() -> None:
    client = FakeQdrantClient()
    node = make_build_conversation_window_node(client, "code_chunks")
    state: AgentState = {"messages": [HumanMessage(content="how does auth work?")]}

    result = node(state)

    assert result == {"conversation_window": [], "conversation_history": ""}
    assert client.retrieve_calls == []


def test_build_conversation_window_node_builds_a_turn_from_the_completed_previous_turn() -> None:
    """messages[-1] is the *current* turn's question (added by the caller before invoke) -
    the completed turn to fold in is the Human/AI pair just before it."""
    client = FakeQdrantClient()
    node = make_build_conversation_window_node(client, "code_chunks")
    state: AgentState = {
        "messages": [
            HumanMessage(content="how does auth work?"),
            AIMessage(content="Auth uses JWT."),
            HumanMessage(content="what about the retry logic?"),
        ]
    }

    result = node(state)

    assert result["conversation_window"] == [
        {"question": "how does auth work?", "answer": "Auth uses JWT.", "cited_context": ""}
    ]
    assert "how does auth work?" in result["conversation_history"]
    assert "Auth uses JWT." in result["conversation_history"]
    assert client.retrieve_calls == []


def test_build_conversation_window_node_resolves_the_previous_answers_citations() -> None:
    client = FakeQdrantClient(retrieve_response=[_make_record("id-1", context_text="used to authenticate")])
    node = make_build_conversation_window_node(client, "code_chunks")
    state: AgentState = {
        "messages": [
            HumanMessage(content="how does auth work?"),
            AIMessage(content="Auth uses JWT.", additional_kwargs={"citations": ["id-1"]}),
            HumanMessage(content="what about the retry logic?"),
        ]
    }

    result = node(state)

    assert client.retrieve_calls[0]["collection_name"] == "code_chunks"
    assert client.retrieve_calls[0]["ids"] == ["id-1"]
    cited_context = result["conversation_window"][0]["cited_context"]
    assert "src/auth.py" in cited_context
    assert "used to authenticate" in cited_context
    assert "src/auth.py" in result["conversation_history"]


def test_build_conversation_window_node_evicts_the_oldest_turn_once_over_the_cap(monkeypatch) -> None:
    import query_agent.nodes as nodes_module

    monkeypatch.setattr(nodes_module, "CONVERSATION_WINDOW_TURNS", 2)
    client = FakeQdrantClient()
    node = make_build_conversation_window_node(client, "code_chunks")
    state: AgentState = {
        "conversation_window": [
            {"question": "q1", "answer": "a1", "cited_context": ""},
            {"question": "q2", "answer": "a2", "cited_context": ""},
        ],
        "messages": [
            HumanMessage(content="q3"),
            AIMessage(content="a3"),
            HumanMessage(content="q4"),
        ],
    }

    result = node(state)

    assert [turn["question"] for turn in result["conversation_window"]] == ["q2", "q3"]


def test_evaluate_question_node_returns_the_structured_fields() -> None:
    result = EvaluateQuestion(
        question_type="implementation",
        synthesized_query="how is auth implemented in the codebase",
        filters=QuestionFilters(language=None),
        expects_multiple_retrievals=False,
    )
    llm = FakeLLM([result])
    node = make_evaluate_question_node(llm)
    state: AgentState = {"question": "how is auth implemented?"}

    output = node(state)

    assert output == {
        "question_type": "implementation",
        "search_query": "how is auth implemented in the codebase",
        "search_filters": {"language": None},
        "expects_multiple_retrievals": False,
    }


def test_evaluate_question_node_folds_conversation_window_into_the_prompt() -> None:
    result = EvaluateQuestion(
        question_type="implementation",
        synthesized_query="what does the retry logic in payments do",
        filters=QuestionFilters(language=None),
        expects_multiple_retrievals=False,
    )
    llm = FakeLLM([result])
    node = make_evaluate_question_node(llm)
    state: AgentState = {
        "question": "no, I meant the retry logic",
        "conversation_history": (
            "Conversation history:\nQ: how does auth work?\nA: Auth uses JWT.\n"
            "Cited code:\nfile_path=src/auth.py start_line=1 end_line=1\nauthenticate:\ndef authenticate(): ...\n\n"
        ),
    }

    node(state)

    prompt = llm.structured.invoke_calls[0][1].content
    assert "how does auth work?" in prompt
    assert "Auth uses JWT." in prompt
    assert "src/auth.py" in prompt
    assert "no, I meant the retry logic" in prompt


def test_retrieve_documents_node_searches_with_search_query_and_filters(monkeypatch) -> None:
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])
    client = FakeQdrantClient(query_response=FakeQueryResponse([_make_point(score=0.9)]))
    node = make_retrieve_documents_node(client, "code_chunks")
    state: AgentState = {"search_query": "how does auth work?", "search_filters": {"language": None}, "retrieved_chunks":{}}

    result = node(state)

    assert result["retrieval_attempts"] == 1
    [chunk] = result["retrieved_chunks"].values()
    assert chunk["file_path"] == "src/auth.py"
    assert chunk["symbol_name"] == "authenticate"
    assert chunk["score"] == 0.9


def test_retrieve_documents_node_increments_retrieval_attempts_from_existing_state(monkeypatch) -> None:
    monkeypatch.setattr(chunks, "embed_text", lambda text: [0.1, 0.2, 0.3])
    client = FakeQdrantClient()
    node = make_retrieve_documents_node(client, "code_chunks")
    state: AgentState = {
        "search_query": "how does auth work?",
        "search_filters": {"language": None},
        "retrieval_attempts": 2,
        "retrieved_chunks":{}
    }

    result = node(state)

    assert result["retrieval_attempts"] == 3


def _conversation_history() -> str:
    return "Conversation history:\nQ: how does auth work?\nA: Auth uses JWT.\n\n"


def test_grade_documents_node_folds_conversation_window_into_the_prompt() -> None:
    llm = FakeLLM([GradeDocument(relevant="yes")])
    node = make_grade_documents_node(llm)
    state: AgentState = {
        "question": "what about the retry logic?",
        "retrieved_chunks": {"id-1": _search_result(id="id-1")},
        "chunk_relevance": {},
        "conversation_history": _conversation_history(),
    }

    node(state)

    prompt = llm.structured.invoke_calls[0][1].content
    assert "how does auth work?" in prompt
    assert "Auth uses JWT." in prompt


def test_grade_documents_node_grades_each_not_yet_graded_chunk() -> None:
    llm = FakeLLM([GradeDocument(relevant="yes"), GradeDocument(relevant="no")])
    node = make_grade_documents_node(llm)
    state: AgentState = {
        "question": "how does auth work?",
        "retrieved_chunks": {
            "id-1": _search_result(id="id-1", file_path="src/auth.py"),
            "id-2": _search_result(id="id-2", file_path="src/unrelated.py"),
        },
        "chunk_relevance": {},
    }

    result = node(state)

    assert result["chunk_relevance"] == {"id-1": "yes", "id-2": "no"}


def test_grade_documents_node_skips_already_graded_chunks() -> None:
    llm = FakeLLM([GradeDocument(relevant="no")])
    node = make_grade_documents_node(llm)
    state: AgentState = {
        "question": "how does auth work?",
        "retrieved_chunks": {"id-1": _search_result(id="id-1"), "id-2": _search_result(id="id-2")},
        "chunk_relevance": {"id-1": "yes"},
    }

    result = node(state)

    assert result["chunk_relevance"] == {"id-1": "yes", "id-2": "no"}


def test_generate_answer_node_uses_only_yes_labeled_chunks() -> None:
    generated = GeneratedAnswer(text="Auth is handled in authenticate().", cited_chunk_ids=["id-1"])
    llm = FakeLLM([generated])
    node = make_generate_answer_node(llm)
    state: AgentState = {
        "question": "how does auth work?",
        "retrieved_chunks": {
            "id-1": _search_result(id="id-1", file_path="src/auth.py", symbol_name="authenticate"),
            "id-2": _search_result(id="id-2", file_path="src/unrelated.py", symbol_name="noop"),
        },
        "chunk_relevance": {"id-1": "yes", "id-2": "no"},
    }

    result = node(state)

    assert result == {
        "answer": Answer(
            text="Auth is handled in authenticate().",
            citations=[Citation(chunk_id="id-1", file_path="src/auth.py", start_line=1, end_line=1, citation_text="def authenticate(): ...")],
        )
    }
    prompt = llm.structured.invoke_calls[0][1].content
    assert "chunk_id=id-1" in prompt
    assert "src/auth.py" in prompt
    assert "src/unrelated.py" not in prompt


def test_generate_answer_node_drops_hallucinated_chunk_ids() -> None:
    generated = GeneratedAnswer(text="Auth is handled in authenticate().", cited_chunk_ids=["id-1", "not-a-real-id"])
    llm = FakeLLM([generated])
    node = make_generate_answer_node(llm)
    state: AgentState = {
        "question": "how does auth work?",
        "retrieved_chunks": {"id-1": _search_result(id="id-1", file_path="src/auth.py", symbol_name="authenticate")},
        "chunk_relevance": {"id-1": "yes"},
    }

    result = node(state)

    assert [c.chunk_id for c in result["answer"].citations] == ["id-1"]


def test_generate_answer_node_folds_conversation_window_into_the_prompt() -> None:
    generated = GeneratedAnswer(text="It also handles images.", cited_chunk_ids=[])
    llm = FakeLLM([generated])
    node = make_generate_answer_node(llm)
    state: AgentState = {
        "question": "what if it's an image?",
        "retrieved_chunks": {},
        "chunk_relevance": {},
        "conversation_history": _conversation_history(),
    }

    node(state)

    prompt = llm.structured.invoke_calls[0][1].content
    assert "how does auth work?" in prompt
    assert "Auth uses JWT." in prompt


def test_evaluate_answer_node_returns_good_grade_without_touching_search_query() -> None:
    result = EvaluateAnswer(grade="good", reasoning="Answer is accurate and complete.")
    llm = FakeLLM([result])
    node = make_evaluate_answer_node(llm)
    state: AgentState = {
        "question": "how does auth work?",
        "search_query": "how is auth implemented",
        "answer": Answer(text="Auth uses JWT.", citations=[]),
    }

    output = node(state)

    assert output == {"answer_grade": "good", "evaluation_reasoning": "Answer is accurate and complete."}


def test_evaluate_answer_node_replaces_search_query_with_revised_query_on_bad_grade() -> None:
    result = EvaluateAnswer(
        grade="bad",
        reasoning="Doesn't mention token expiry.",
        revised_search_query="token expiry handling in auth",
    )
    llm = FakeLLM([result])
    node = make_evaluate_answer_node(llm)
    state: AgentState = {
        "question": "how does auth work?",
        "search_query": "how is auth implemented",
        "answer": Answer(text="Auth uses JWT.", citations=[]),
    }

    output = node(state)

    assert output["answer_grade"] == "bad"
    assert output["search_query"] == "token expiry handling in auth"


def test_evaluate_answer_node_folds_conversation_window_into_the_prompt() -> None:
    result = EvaluateAnswer(grade="good", reasoning="Resolves the follow-up.")
    llm = FakeLLM([result])
    node = make_evaluate_answer_node(llm)
    state: AgentState = {
        "question": "what if it's an image?",
        "search_query": "does image upload get the same invalid-payload handling",
        "answer": Answer(text="It also handles images.", citations=[]),
        "conversation_history": _conversation_history(),
    }

    node(state)

    prompt = llm.structured.invoke_calls[0][1].content
    assert "how does auth work?" in prompt
    assert "Auth uses JWT." in prompt
