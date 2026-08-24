from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.postgres import PostgresSaver
from db.db_postgres import ensure_checkpointer_tables
from query_agent.schemas import Answer
from query_agent.effort import BasicEffort, MediumEffort, HighEffort
from psycopg_pool import ConnectionPool


def init_checkpointer(pool: ConnectionPool) -> PostgresSaver:
    """Build a ready-to-use PostgresSaver over `pool`: LangGraph's conversation-state checkpointer, keyed by thread_id.

    Runs the checkpointer's own (idempotent) schema setup before returning it.
    """
    serde = JsonPlusSerializer(allowed_msgpack_modules=[BasicEffort, Answer, MediumEffort, HighEffort])
    checkpointer = PostgresSaver(conn=pool, serde=serde)
    ensure_checkpointer_tables(pool)
    return checkpointer
