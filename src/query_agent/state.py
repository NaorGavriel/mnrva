from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from models import ChunkSearchResult
from query_agent.agent_schemas import Answer
from query_agent.effort import Effort


def retrieved_chunks_reducer(existing: dict[str, ChunkSearchResult], new: dict[str, ChunkSearchResult]) -> dict[str, ChunkSearchResult]:
    """Merge id-keyed chunk dicts across retrieval attempts; new hits take precedence on id collision."""
    return {**existing, **new}


class AgentState(TypedDict):
    """LangGraph state for one query-agent turn: the question, its evaluation, retrieved chunks, and the generated answer."""

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    effort: Effort
    question_type: Literal["implementation", "architecture", "heuristic", "symbol", "workflow"]
    search_query: str
    search_filters: dict
    expects_multiple_retrievals: bool
    retrieved_chunks: Annotated[dict[str, ChunkSearchResult], retrieved_chunks_reducer]
    chunk_relevance: dict[str, Literal["yes", "no"]]
    retrieval_attempts: int
    answer: Answer | None
    answer_grade: Literal["good", "bad"] | None
    evaluation_reasoning: str | None
