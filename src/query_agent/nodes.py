from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from qdrant_client import QdrantClient

from chunks import search_chunks
from query_agent.agent_prompts import EVALUATE_QUESTION_SYSTEM_PROMPT, GRADE_DOCUMENT_SYSTEM_PROMPT
from query_agent.agent_schemas import EvaluateQuestion, GradeDocument
from query_agent.state import AgentState


def make_evaluate_question_node(llm: Runnable[Any, AIMessage]) -> Callable[[AgentState], dict]:
    """Build the `evaluate_question` node: classifies the question and produces a synthesized search query + filters."""
    structured_llm = llm.with_structured_output(EvaluateQuestion)

    def evaluate_question_node(state: AgentState) -> dict:
        result: EvaluateQuestion = structured_llm.invoke(
            [SystemMessage(EVALUATE_QUESTION_SYSTEM_PROMPT), HumanMessage(state["question"])]
        )
        
        return {
            "question_type": result.question_type,
            "search_query": result.synthesized_query,
            "search_filters": result.filters.model_dump(),
            "expects_multiple_retrievals": result.expects_multiple_retrievals,
        }

    return evaluate_question_node


def make_retrieve_documents_node(client: QdrantClient, collection_name: str) -> Callable[[AgentState], dict]:
    """Build the `retrieve_documents` node, bound to a Qdrant `client`/`collection_name`."""

    def retrieve_documents_node(state: AgentState) -> dict:
        filters = state["search_filters"]
        chunks = search_chunks(
            client,
            collection_name,
            state["search_query"],
            language=filters.get("language"),
            kind=filters.get("kind"),
        )
        return {"retrieved_chunks": chunks}

    return retrieve_documents_node


def make_grade_documents_node(llm: Runnable[Any, AIMessage]) -> Callable[[AgentState], dict]:
    """Build the `grade_documents` node: labels each not-yet-graded retrieved chunk yes/no for relevance to the question."""
    structured_llm = llm.with_structured_output(GradeDocument)

    def grade_documents_node(state: AgentState) -> dict:
        chunk_relevance = dict(state.get("chunk_relevance", {}))
        for chunk in state["retrieved_chunks"]:
            if chunk["id"] in chunk_relevance:
                continue
            prompt = (
                f"User's question:\n{state['question']}\n\n"
                f"Retrieved chunk ({chunk['file_path']}, {chunk['symbol_name']}):\n"
                f"{chunk['context_text'] or ''}\n\n{chunk['raw_text']}"
            )
            result: GradeDocument = structured_llm.invoke(
                [SystemMessage(GRADE_DOCUMENT_SYSTEM_PROMPT), HumanMessage(prompt)]
            )
            chunk_relevance[chunk["id"]] = result.relevant
        return {"chunk_relevance": chunk_relevance}

    return grade_documents_node


def make_generate_answer_node(llm: Runnable[Any, AIMessage]) -> Callable[[AgentState], dict]:
    """Build the `generate_answer` node. `llm` is a plain chat-model Runnable - no structured output yet."""

    def generate_answer_node(state: AgentState) -> dict:
        relevant_chunks = [
            chunk for chunk in state["retrieved_chunks"] if state["chunk_relevance"].get(chunk["id"]) == "yes"
        ]
        context = "\n\n".join(
            f"{chunk['file_path']} ({chunk['symbol_name']}):\n{chunk['raw_text']}"
            for chunk in relevant_chunks
        )
        prompt = (
            f"User's question:\n{state['question']}\n\n"
            f"Relevant code:\n{context}\n\n"
            "Answer the user's question using the code above."
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}

    return generate_answer_node
