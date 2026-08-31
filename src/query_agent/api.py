import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Literal

from dotenv import load_dotenv

load_dotenv()

import db.db_qdrant as db_qdrant
from db.db_postgres import init_async_pool
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from query_agent.effort import BasicEffort, Effort, HighEffort, MediumEffort
from query_agent.graph import build_graph
from query_agent.schemas import Citation
from repo_metadata import aget_repo_metadata

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
    await db_qdrant.wait_until_ready(client)
    pool = await init_async_pool()
    app.state.pool = pool
    app.state.graph = await build_graph(client, pool)
    try:
        yield
    finally:
        await client.close()
        await pool.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Body for one query-agent turn: the question, and an optional effort level."""

    question: str
    effort: Literal["basic", "medium", "high"] = "medium"


class ThreadResponse(BaseModel):
    """Response for POST /threads: the freshly minted thread_id."""

    thread_id: str


class RepoMetadataResponse(BaseModel):
    """Response for GET /repo: the tracked repo's identity and last sync time."""

    github_url: str
    commit_sha: str
    updated_at: datetime


def _format_sse(node_name: str, payload: dict) -> str:
    """Format one SSE message: `node_name` as the event name, `payload` as JSON data."""
    return f"event: {node_name}\ndata: {json.dumps(payload)}\n\n"


def _project_update(node_name: str, node_output: dict, citations: list[Citation]) -> dict:
    """Build the client-facing payload for one node's update.
    `persist_agent_message` carries the final answer text and  citations"""

    if node_name == "persist_agent_message":
        message = node_output["messages"]
        return {
            "label": _STEP_LABELS[node_name],
            "answer": message.content,
            "citations": [citation.model_dump() for citation in citations],
        }
    return {"label": _STEP_LABELS.get(node_name, node_name)}


async def _stream_turn(graph: CompiledStateGraph, thread_id: str, question: str, effort: Effort) -> AsyncIterator[str]:
    """Run one turn through `graph`, yielding one SSE message per finished node.
    Ends with an `error` event instead of `persist_agent_message` if the turn raises."""
    state = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "effort": effort,
        "retrieved_chunks": {},
        "chunk_relevance": {},
        "retrieval_attempts": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}
    citations: list[Citation] = []
    try:
        async for update in graph.astream(state, config, stream_mode="updates"):
            node_name, node_output = next(iter(update.items()))
            if node_name == "generate_answer":
                citations = node_output["answer"].citations
            yield _format_sse(node_name, _project_update(node_name, node_output, citations))
    except Exception as exc:
        yield _format_sse("error", {"message": str(exc)})


@app.post("/threads")
async def create_thread() -> ThreadResponse:
    """Mint a new thread_id for a fresh conversation."""
    return ThreadResponse(thread_id=str(uuid.uuid4()))


@app.get("/repo")
async def get_repo() -> RepoMetadataResponse:
    """Return the tracked repo's github_url/commit_sha/last-synced timestamp."""
    metadata = await aget_repo_metadata(app.state.pool)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Repo has not been ingested yet")
    return RepoMetadataResponse(**metadata)


@app.post("/threads/{thread_id}/query")
async def query(thread_id: str, request: QueryRequest) -> StreamingResponse:
    """Stream one query-agent turn's per-node progress as SSE, ending with the final answer."""
    effort = _EFFORT_BY_NAME[request.effort]()
    return StreamingResponse(
        _stream_turn(app.state.graph, thread_id, request.question, effort),
        media_type="text/event-stream",
    )
