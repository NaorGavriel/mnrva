from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from models import ChunkSearchResult
from query_agent.schemas import Answer
from query_agent.effort import Effort


class TurnContext(TypedDict):
    """One past conversation turn - question, answer, and the content of any chunks it
    cited (resolved from citation ids, not just their ids) - kept in `conversation_window`."""

    question: str
    answer: str
    cited_context: str


class AgentState(TypedDict):
    """LangGraph state for one query-agent turn: the question, its evaluation, retrieved chunks, and the generated answer."""

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    conversation_window: list[TurnContext]
    conversation_history: str
    effort: Effort
    question_type: Literal["implementation", "architecture", "heuristic", "symbol", "workflow"]
    search_query: str
    search_filters: dict
    expects_multiple_retrievals: bool
    retrieved_chunks: dict[str, ChunkSearchResult]
    chunk_relevance: dict[str, Literal["yes", "no"]]
    retrieval_attempts: int
    answer: Answer | None
    answer_grade: Literal["good", "bad"] | None
    evaluation_reasoning: str | None
