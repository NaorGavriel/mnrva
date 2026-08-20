from typing import Literal

from pydantic import BaseModel, Field


class QuestionFilters(BaseModel):
    """Payload filters to scope retrieval, extracted from the user's question."""

    language: str | None = Field(default=None, description="Programming language named or implied by the question, if any.")
    # Must match Chunk.kind (models.py) - an invalid value silently zeroes out the Qdrant filter match.
    kind: Literal["class", "function", "section"] | None = Field(default=None, description="Chunk kind named or implied by the question, if any.")


class EvaluateQuestion(BaseModel):
    """Structured output of the `evaluate_question` node."""

    question_type: Literal["implementation", "architecture", "heuristic", "symbol", "workflow"]
    synthesized_query: str = Field(description="The question plus any added context - what actually gets searched, not the raw question.")
    filters: QuestionFilters
    expects_multiple_retrievals: bool = Field(description="Whether answering this question likely requires more than one retrieval attempt.")


class GradeDocument(BaseModel):
    """Structured output of grading one retrieved chunk's relevance to the user's question."""

    relevant: Literal["yes", "no"]


class Citation(BaseModel):
    """A source-code location backing a claim in the agent's answer."""

    file_path: str
    start_line: int
    end_line: int
    citation_text: str # could be code or free form text if it's from .md for example


class GrepMatch(BaseModel):
    """One matching line from `grep_search_tool`."""

    file_path: str
    line_number: int
    line_text: str