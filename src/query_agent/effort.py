from pydantic import BaseModel


class Effort(BaseModel):
    """Levers that scale retrieval thoroughness for one query-agent turn."""

    retrieval_attempts_cap: int
    max_tool_calls: int


class BasicEffort(Effort):
    """Minimal effort: no corrective retry, a small tool-call budget."""

    retrieval_attempts_cap: int = 1
    max_tool_calls: int = 3


class MediumEffort(Effort):
    """Default effort: a few corrective retries, a moderate tool-call budget."""

    retrieval_attempts_cap: int = 3
    max_tool_calls: int = 5


class HighEffort(Effort):
    """Maximum effort: the full corrective-retry and tool-call budget."""

    retrieval_attempts_cap: int = 5
    max_tool_calls: int = 10
