from typing import Literal

from pydantic import BaseModel, Field


class QuestionFilters(BaseModel):
    """Payload filters to scope retrieval, extracted from the user's question."""

    language: str | None = Field(default=None, description="Programming language named or implied by the question, if any.")
    # Must match Chunk.kind (models.py) - an invalid value silently zeroes out the Qdrant filter match.


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
    # None for prose-section chunks, which don't carry a line range (prose_parser.py).
    start_line: int | None
    end_line: int | None
    citation_text: str # could be code or free form text if it's from .md for example


class Answer(BaseModel):
    """Structured output of the `generate_answer` node: the answer text plus its backing citations."""

    text: str
    citations: list[Citation] = Field(description="At least one citation backing every claim made in text.")


class EvaluateAnswer(BaseModel):
    """Structured output of grading the generated Answer against the original user question."""

    grade: Literal["good", "bad"]
    reasoning: str = Field(
        description="Why the answer does or doesn't hold up against the question - "
        "appended to search_query on a bad grade to give the retry a concrete reason to look different."
    )