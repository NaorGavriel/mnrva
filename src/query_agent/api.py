import json
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from dotenv import load_dotenv

load_dotenv()

import db.db_qdrant as db_qdrant
from db.db_postgres import init_async_pool
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from query_agent.effort import BasicEffort, Effort, HighEffort, MediumEffort
from query_agent.graph import build_graph

_EFFORT_BY_NAME: dict[str, type[Effort]] = {"basic": BasicEffort, "medium": MediumEffort, "high": HighEffort}

_STEP_LABELS = {
    "build_conversation_window": "Preparing conversation context",
    "evaluate_question": "Understanding the question",
    "retrieve_documents": "Searching the codebase",
    "grade_documents": "Grading retrieved results",
    "generate_answer": "Drafting an answer",
    "evaluate_answer": "Checking answer quality",
    "persist_agent_message": "Done",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the Qdrant client and Postgres pool once for the app's lifetime, and build the graph
    once from them - it's reused across requests, threads distinguished by thread_id."""
    client = db_qdrant.init_async_client(url=db_qdrant.QDRANT_URL, api_key=db_qdrant.QDRANT_API_KEY)
    pool = await init_async_pool()
    app.state.graph = await build_graph(client, pool)
    try:
        yield
    finally:
        await client.close()
        await pool.close()


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    """Body for one query-agent turn: the question, and an optional effort level."""

    question: str
    effort: Literal["basic", "medium", "high"] = "medium"


def _format_sse(node_name: str, payload: dict) -> str:
    """Format one SSE message: `node_name` as the event name, `payload` as JSON data."""
    return f"event: {node_name}\ndata: {json.dumps(payload)}\n\n"


def _project_update(node_name: str, node_output: dict) -> dict:
    """Build the client-facing payload for one node's update.
    `persist_agent_message` additionally carries the final answer text and citation ids since the graph reached the end."""

    if node_name == "persist_agent_message":
        message = node_output["messages"]
        return {
            "label": _STEP_LABELS[node_name],
            "answer": message.content,
            "citations": message.additional_kwargs.get("citations", []),
        }
    return {"label": _STEP_LABELS.get(node_name, node_name)}


async def _stream_turn(graph: CompiledStateGraph, thread_id: str, question: str, effort: Effort) -> AsyncIterator[str]:
    """Run one turn through `graph`, yielding one SSE message per finished node."""
    state = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "effort": effort,
        "retrieved_chunks": {},
        "chunk_relevance": {},
        "retrieval_attempts": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}
    async for update in graph.astream(state, config, stream_mode="updates"):
        node_name, node_output = next(iter(update.items()))
        yield _format_sse(node_name, _project_update(node_name, node_output))


@app.post("/threads/{thread_id}/query")
async def query(thread_id: str, request: QueryRequest) -> StreamingResponse:
    """Stream one query-agent turn's per-node progress as SSE, ending with the final answer."""
    effort = _EFFORT_BY_NAME[request.effort]()
    return StreamingResponse(
        _stream_turn(app.state.graph, thread_id, request.question, effort),
        media_type="text/event-stream",
    )
