from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """LangGraph state for one query-agent conversation, keyed by thread_id (`docs/query_agent.md` §2.4)."""

    messages: Annotated[list[AnyMessage], add_messages]
    answer_grade: Literal["good", "bad"] | None
    evaluation_reasoning: str | None
    retry_count: int
