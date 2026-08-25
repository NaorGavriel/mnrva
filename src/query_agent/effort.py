from pydantic import BaseModel


class Effort(BaseModel):
    """Levers that scale retrieval thoroughness for one query-agent turn."""

    retrieval_attempts_cap: int


class BasicEffort(Effort):
    """Minimal effort: no corrective retry."""

    retrieval_attempts_cap: int = 0


class MediumEffort(Effort):
    """Default effort: a few corrective retries."""

    retrieval_attempts_cap: int = 2


class HighEffort(Effort):
    """Maximum effort: the full corrective-retry."""

    retrieval_attempts_cap: int = 4
