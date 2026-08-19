from typing import Any, Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from qdrant_client import QdrantClient

from chunks import search_chunks
from query_agent.state import AgentState


def make_retrieve_documents_node(client: QdrantClient, collection_name: str) -> Callable[[AgentState], dict]:
    """Build the `retrieve_documents` node, bound to a Qdrant `client`/`collection_name`."""

    def retrieve_documents_node(state: AgentState) -> dict:
        chunks = search_chunks(client, collection_name, state["question"])
        return {"retrieved_chunks": chunks}

    return retrieve_documents_node


def make_generate_answer_node(llm: Runnable[Any, AIMessage]) -> Callable[[AgentState], dict]:
    """Build the `generate_answer` node. `llm` is a plain chat-model Runnable - no structured output yet."""

    def generate_answer_node(state: AgentState) -> dict:
        context = "\n\n".join(
            f"{chunk['file_path']} ({chunk['symbol_name']}):\n{chunk['raw_text']}"
            for chunk in state["retrieved_chunks"]
        )
        prompt = (
            f"User's question:\n{state['question']}\n\n"
            f"Relevant code:\n{context}\n\n"
            "Answer the user's question using the code above."
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}

    return generate_answer_node
