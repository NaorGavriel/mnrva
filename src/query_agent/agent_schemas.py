from pydantic import BaseModel, Field

class Citation(BaseModel):
    """A source-code location backing a claim in the agent's answer."""

    file_path: str
    start_line: int
    end_line: int
    citation_text: str # could be code or free form text if it's from .md for example