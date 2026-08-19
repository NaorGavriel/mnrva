from typing import TypedDict

from models import ChunkSearchResult


class AgentState(TypedDict):
    """LangGraph state for one query-agent turn: the question, its retrieved chunks, and the generated answer."""

    question: str
    retrieved_chunks: list[ChunkSearchResult]
    answer: str
