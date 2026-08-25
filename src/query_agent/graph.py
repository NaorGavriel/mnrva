import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from query_agent.checkpointer_postgres import init_checkpointer
from db.db_postgres import init_pool

from db.db_qdrant import COLLECTION_NAME, QDRANT_URL, init_client
from query_agent.nodes import (
    make_build_conversation_window_node,
    make_evaluate_answer_node,
    make_evaluate_question_node,
    make_generate_answer_node,
    make_grade_documents_node,
    make_retrieve_documents_node,
    make_persist_agent_message_node
)
from query_agent.state import AgentState

load_dotenv()

def _route_after_evaluate_answer(state: AgentState) -> str:
    """Loop back to retrieve_documents on a bad grade under the attempt cap; otherwise end the turn."""
    effort = state["effort"]
    
    if state["answer_grade"] == "bad" and state["retrieval_attempts"] < effort.retrieval_attempts_cap:
        return "retrieve_documents"
    return "persist_agent_message"


def build_graph() -> CompiledStateGraph:
    """Assemble the query agent's graph: build_conversation_window -> evaluate_question -> retrieve_documents ->
    grade_documents -> generate_answer -> evaluate_answer, looping back to retrieve_documents on a bad grade."""
    client = init_client(url=QDRANT_URL)
    llm = ChatOpenAI(model=os.environ["AGENT_MODEL"])

    # nodes
    graph = StateGraph(AgentState)
    graph.add_node("build_conversation_window", make_build_conversation_window_node(client, COLLECTION_NAME))
    graph.add_node("evaluate_question", make_evaluate_question_node(llm))
    graph.add_node("retrieve_documents", make_retrieve_documents_node(client, COLLECTION_NAME))
    graph.add_node("grade_documents", make_grade_documents_node(llm))
    graph.add_node("generate_answer", make_generate_answer_node(llm))
    graph.add_node("evaluate_answer", make_evaluate_answer_node(llm))
    graph.add_node("persist_agent_message", make_persist_agent_message_node())
    
    # edges
    graph.add_edge(START, "build_conversation_window")
    graph.add_edge("build_conversation_window", "evaluate_question")
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
    pool = init_pool()
    checkpointer = init_checkpointer(pool=pool)


    return graph.compile(checkpointer=checkpointer)
