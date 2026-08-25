import os
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from qdrant_client import QdrantClient

from chunks import get_chunks_by_id, search_chunks
from query_agent.agent_prompts import (
    EVALUATE_ANSWER_SYSTEM_PROMPT,
    EVALUATE_QUESTION_SYSTEM_PROMPT,
    GENERATE_ANSWER_SYSTEM_PROMPT,
    GRADE_DOCUMENT_SYSTEM_PROMPT
)
from dotenv import load_dotenv
from query_agent.schemas import Answer, Citation, EvaluateAnswer, EvaluateQuestion, GeneratedAnswer, GradeDocument
from query_agent.state import AgentState, TurnContext

load_dotenv()
GRADE_MAX_CONCURRENCY = int(os.environ["GRADE_MAX_CONCURRENCY"])
CONVERSATION_WINDOW_TURNS = int(os.environ["CONVERSATION_WINDOW_TURNS"])
TOP_K = int(os.environ["TOP_CHUNKS_TO_RETREIVE"])

def _format_turn(turn: TurnContext) -> str:
    block = f"Q: {turn['question']}\nA: {turn['answer']}"
    if turn["cited_context"]:
        block += f"\nCited code:\n{turn['cited_context']}"
    return block


def _format_conversation_history(conversation_window: list[TurnContext]) -> str:
    """Format `conversation_window` as a `Conversation history:` block to prepend to a
    node's own prompt, or "" when there's no prior history yet."""
    history = "\n\n".join(_format_turn(turn) for turn in conversation_window)
    return f"Conversation history:\n{history}\n\n" if history else ""


def make_build_conversation_window_node(client: QdrantClient, collection_name: str) -> Callable[[AgentState], dict]:
    """Build the `build_conversation_window` node, bound to a Qdrant `client`/`collection_name`.

    Turns the previous turn into one `conversation_window` entry - question, answer, and the
    current content of any chunks it cited - appends it to the existing window, and trims to
    the most recent `CONVERSATION_WINDOW_TURNS`. Also formats the result into
    `conversation_history` once here, so the other three LLM nodes just read that instead of
    each reformatting `conversation_window` themselves. No-op on the first turn.
    """

    def build_conversation_window_node(state: AgentState) -> dict:
        messages = state.get("messages", [])
        window = list(state.get("conversation_window", []))

        if len(messages) >= 3:  # there's a completed previous turn to fold in
            prev_question, prev_answer = messages[-3], messages[-2]
            citation_ids = prev_answer.additional_kwargs.get("citations", [])
            cited_chunks = get_chunks_by_id(client, collection_name, citation_ids) if citation_ids else []
            cited_context = "\n\n".join(
                f"file_path={chunk.path.as_posix()} start_line={chunk.start_line} end_line={chunk.end_line}\n"
                f"{chunk.symbol_name}:\n{chunk.context_text}"
                for chunk in cited_chunks
            )
            turn: TurnContext = {
                "question": prev_question.content,
                "answer": prev_answer.content,
                "cited_context": cited_context,
            }
            window = (window + [turn])[-CONVERSATION_WINDOW_TURNS:]

        return {
            "conversation_window": window,
            "conversation_history": _format_conversation_history(window),
        }

    return build_conversation_window_node


def make_evaluate_question_node(llm: Runnable[Any, AIMessage]) -> Callable[[AgentState], dict]:
    """Build the `evaluate_question` node: classifies the question and produces a synthesized search query + filters."""
    structured_llm = llm.with_structured_output(EvaluateQuestion)

    def evaluate_question_node(state: AgentState) -> dict:
        question_prompt = f"{state.get('conversation_history', '')}{state['question']}"

        result: EvaluateQuestion = structured_llm.invoke(
            [SystemMessage(EVALUATE_QUESTION_SYSTEM_PROMPT), HumanMessage(question_prompt)]
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
            top_k=TOP_K
        )
        merged_chunks = {**state["retrieved_chunks"], **{r["id"]: r for r in chunks}}
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

        history_prefix = state.get("conversation_history", "")
        prompts = [
            [
                SystemMessage(GRADE_DOCUMENT_SYSTEM_PROMPT),
                HumanMessage(
                    f"{history_prefix}"
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
    """Build the `generate_answer` node: yes-labeled chunks + question -> a structured `Answer` with citations.

    The LLM only produces the answer text and the chunk_ids it drew from; citation metadata
    (file_path, line range, text) is resolved locally from the already-retrieved chunks.
    """
    structured_llm = llm.with_structured_output(GeneratedAnswer)

    def generate_answer_node(state: AgentState) -> dict:
        relevant_chunks = [chunk for chunk in state["retrieved_chunks"].values() if state["chunk_relevance"].get(chunk["id"]) == "yes"]
        chunks_by_id = {chunk["id"]: chunk for chunk in relevant_chunks}

        context = "\n\n".join(
            f"chunk_id={chunk['id']} file_path={chunk['file_path']} start_line={chunk['start_line']} end_line={chunk['end_line']}\n"
            f"{chunk['symbol_name']}:\n{chunk['raw_text']}"
            for chunk in relevant_chunks
        )
        prompt = f"{state.get('conversation_history', '')}User's question:\n{state['question']}\n\nRetrieved chunks:\n{context}"

        generated: GeneratedAnswer = structured_llm.invoke(
            [SystemMessage(GENERATE_ANSWER_SYSTEM_PROMPT), HumanMessage(prompt)]
        )

        citations = [
            Citation(
                chunk_id=chunk_id,
                file_path=chunks_by_id[chunk_id]["file_path"],
                start_line=chunks_by_id[chunk_id]["start_line"],
                end_line=chunks_by_id[chunk_id]["end_line"],
                citation_text=chunks_by_id[chunk_id]["raw_text"],
            )
            for chunk_id in generated.cited_chunk_ids
            if chunk_id in chunks_by_id  # drop any hallucinated/unknown chunk_id
        ]
        return {"answer": Answer(text=generated.text, citations=citations)}

    return generate_answer_node


def make_evaluate_answer_node(llm: Runnable[Any, AIMessage]) -> Callable[[AgentState], dict]:
    """Build the `evaluate_answer` node: grades the generated Answer against the original question,
    replacing search_query with a revised, standalone query on a bad grade to drive re-retrieval."""
    structured_llm = llm.with_structured_output(EvaluateAnswer)

    def evaluate_answer_node(state: AgentState) -> dict:
        answer = state["answer"]
        citations = "\n".join(
            f"- {citation.file_path}:{citation.start_line}-{citation.end_line}"
            for citation in answer.citations
        )
        prompt = (
            f"{state.get('conversation_history', '')}"
            f"User's question:\n{state['question']}\n\n"
            f"Generated answer:\n{answer.text}\n\n"
            f"Citations:\n{citations}"
        )
        result: EvaluateAnswer = structured_llm.invoke(
            [SystemMessage(EVALUATE_ANSWER_SYSTEM_PROMPT), HumanMessage(prompt)]
        )
        answer_grading: dict = {"answer_grade": result.grade, "evaluation_reasoning": result.reasoning}
        if result.grade == "bad":
            answer_grading["search_query"] = result.revised_search_query
        return answer_grading

    return evaluate_answer_node



def make_persist_agent_message_node() -> Callable[[AgentState], dict]:
    def persist_agent_message_node(state: AgentState) -> dict:
        agent_message = AIMessage(content=state["answer"].text,
                                  additional_kwargs={"citations": [c.chunk_id for c in state["answer"].citations]})

        return {"messages": agent_message}

    return persist_agent_message_node