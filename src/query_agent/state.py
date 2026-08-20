from typing import Literal, TypedDict

from models import ChunkSearchResult


class AgentState(TypedDict):
    """LangGraph state for one query-agent turn: the question, its evaluation, retrieved chunks, and the generated answer."""

    question: str
    question_type: Literal["implementation", "architecture", "heuristic", "symbol", "workflow"]
    search_query: str
    search_filters: dict
    expects_multiple_retrievals: bool
    retrieved_chunks: list[ChunkSearchResult]
    answer: str
