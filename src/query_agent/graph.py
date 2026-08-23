import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from query_agent.schemas import Answer
from query_agent.effort import BasicEffort, MediumEffort, HighEffort

from db.db_qdrant import COLLECTION_NAME, QDRANT_URL, init_client
from query_agent.nodes import (
    make_evaluate_answer_node,
    make_evaluate_question_node,
    make_generate_answer_node,
    make_grade_documents_node,
    make_retrieve_documents_node,
    make_persist_agent_message_node
)
from query_agent.state import AgentState

load_dotenv()

RETRIEVAL_ATTEMPTS_CAP = 3


def _route_after_evaluate_answer(state: AgentState) -> str:
    """Loop back to retrieve_documents on a bad grade under the attempt cap; otherwise end the turn."""
    if state["answer_grade"] == "bad" and state["retrieval_attempts"] < RETRIEVAL_ATTEMPTS_CAP:
        return "retrieve_documents"
    return "persist_agent_message"


def build_graph() -> CompiledStateGraph:
    """Assemble the query agent's graph: evaluate_question -> retrieve_documents -> grade_documents ->
    generate_answer -> evaluate_answer, looping back to retrieve_documents on a bad grade."""
    client = init_client(url=QDRANT_URL)
    llm = ChatOpenAI(model=os.environ["AGENT_MODEL"])

    # nodes
    graph = StateGraph(AgentState)
    graph.add_node("evaluate_question", make_evaluate_question_node(llm))
    graph.add_node("retrieve_documents", make_retrieve_documents_node(client, COLLECTION_NAME))
    graph.add_node("grade_documents", make_grade_documents_node(llm))
    graph.add_node("generate_answer", make_generate_answer_node(llm))
    graph.add_node("evaluate_answer", make_evaluate_answer_node(llm))
    graph.add_node("persist_agent_message", make_persist_agent_message_node())
    
    # edges
    graph.add_edge(START, "evaluate_question")
    graph.add_edge("evaluate_question", "retrieve_documents")
    graph.add_edge("retrieve_documents", "grade_documents")
    graph.add_edge("grade_documents", "generate_answer")
    graph.add_edge("generate_answer", "evaluate_answer")
    graph.add_conditional_edges(
        "evaluate_answer", _route_after_evaluate_answer, {"retrieve_documents": "retrieve_documents",
                                                            "persist_agent_message": "persist_agent_message"}
    )
    graph.add_edge("persist_agent_message", END)

    # checkpointer
    serde = JsonPlusSerializer(allowed_msgpack_modules=[BasicEffort, Answer, MediumEffort, HighEffort])
    checkpointer = InMemorySaver(serde=serde)

    return graph.compile(checkpointer=checkpointer)
