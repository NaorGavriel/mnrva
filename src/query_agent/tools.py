from langchain_core.tools import BaseTool, tool
from qdrant_client import QdrantClient

from chunks import get_chunks_by_id as _get_chunks_by_id


def make_get_chunks_by_id_tool(client: QdrantClient, collection_name: str) -> BaseTool:
    """Build the `get_chunks_by_id` tool, bound to a Qdrant `client`/`collection_name`."""

    @tool
    def get_chunks_by_id(chunk_ids: list[str]) -> str:
        """Fetch the current content of chunks by id to supply a follow-up question with relevant context."""
        chunks = _get_chunks_by_id(client, collection_name, chunk_ids)
        found_ids = {chunk.id for chunk in chunks}
        parts = [
            f"file_path={chunk.path.as_posix()} start_line={chunk.start_line} end_line={chunk.end_line}\n"
            f"{chunk.symbol_name}:\n{chunk.raw_text}"
            for chunk in chunks
        ]
        missing_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in found_ids]
        if missing_ids:
            parts.append(f"Not found (the code may have changed since): {', '.join(missing_ids)}")
        return "\n\n".join(parts)

    return get_chunks_by_id
