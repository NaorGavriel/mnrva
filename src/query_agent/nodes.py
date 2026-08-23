import os
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from qdrant_client import QdrantClient

from chunks import search_chunks
from query_agent.agent_prompts import (
    EVALUATE_ANSWER_SYSTEM_PROMPT,
    EVALUATE_QUESTION_SYSTEM_PROMPT,
    GENERATE_ANSWER_SYSTEM_PROMPT,
    GRADE_DOCUMENT_SYSTEM_PROMPT
)
from dotenv import load_dotenv
from query_agent.schemas import Answer, EvaluateAnswer, EvaluateQuestion, GradeDocument
from query_agent.state import AgentState

load_dotenv()
GRADE_MAX_CONCURRENCY = int(os.environ["GRADE_MAX_CONCURRENCY"])

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
        merged_chunks = {**state["retrieved_chunks"], **{r.id: r for r in chunks}}
        return {
            "retrieved_chunks": merged_chunks,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    return retrieve_documents_node


def make_grade_documents_node(llm: Runnable[Any, AIMessage]) -> Callable[[AgentState], dict]:
    """Build the `grade_documents` node: labels each not-yet-graded retrieved chunk yes/no for relevance to the question."""
    structured_llm = llm.with_structured_output(GradeDocument)

    def grade_documents_node(state: AgentState) -> dict:
        chunk_relevance = dict(state.get("chunk_relevance", {}))
        ungraded = [chunk for chunk in state["retrieved_chunks"].values() if chunk["id"] not in chunk_relevance]
        if not ungraded:
            return {"chunk_relevance": chunk_relevance}

        prompts = [
            [
                SystemMessage(GRADE_DOCUMENT_SYSTEM_PROMPT),
                HumanMessage(
                    f"User's question:\n{state['question']}\n\n"
                    f"Retrieved chunk ({chunk['file_path']}, {chunk['symbol_name']}):\n"
                    f"{chunk['context_text'] or ''}\n\n{chunk['raw_text']}"
                ),
            ]
            for chunk in ungraded
        ]
        results: list[GradeDocument] = structured_llm.batch(prompts, config={"max_concurrency": GRADE_MAX_CONCURRENCY})
        for chunk, result in zip(ungraded, results):
            chunk_relevance[chunk["id"]] = result.relevant
        return {"chunk_relevance": chunk_relevance}

    return grade_documents_node


def make_generate_answer_node(llm: Runnable[Any, AIMessage]) -> Callable[[AgentState], dict]:
    """Build the `generate_answer` node: yes-labeled chunks + question -> a structured `Answer` with citations."""
    structured_llm = llm.with_structured_output(Answer)

    def generate_answer_node(state: AgentState) -> dict:
        relevant_chunks = [chunk for chunk in state["retrieved_chunks"].values() if state["chunk_relevance"].get(chunk["id"]) == "yes"]
        
        context = "\n\n".join(
            f"file_path={chunk['file_path']} start_line={chunk['start_line']} end_line={chunk['end_line']}\n"
            f"{chunk['symbol_name']}:\n{chunk['raw_text']}"
            for chunk in relevant_chunks
        )
        prompt = f"User's question:\n{state['question']}\n\nRetrieved chunks:\n{context}"
        answer: Answer = structured_llm.invoke(
            [SystemMessage(GENERATE_ANSWER_SYSTEM_PROMPT), HumanMessage(prompt)]
        )
        return {"answer": answer}

    return generate_answer_node


def make_evaluate_answer_node(llm: Runnable[Any, AIMessage]) -> Callable[[AgentState], dict]:
    """Build the `evaluate_answer` node: grades the generated Answer against the original question,
    appending its reasoning to search_query on a bad grade to drive re-retrieval."""
    structured_llm = llm.with_structured_output(EvaluateAnswer)

    def evaluate_answer_node(state: AgentState) -> dict:
        answer = state["answer"]
        citations = "\n".join(
            f"- {citation.file_path}:{citation.start_line}-{citation.end_line}: {citation.citation_text}"
            for citation in answer.citations
        )
        prompt = (
            f"User's question:\n{state['question']}\n\n"
            f"Generated answer:\n{answer.text}\n\n"
            f"Citations:\n{citations}"
        )
        result: EvaluateAnswer = structured_llm.invoke(
            [SystemMessage(EVALUATE_ANSWER_SYSTEM_PROMPT), HumanMessage(prompt)]
        )
        updates: dict = {"answer_grade": result.grade, "evaluation_reasoning": result.reasoning}
        if result.grade == "bad":
            updates["search_query"] = f"{state['search_query']}\n\nPrevious attempt's answer fell short: {result.reasoning}"
        return updates

    return evaluate_answer_node
