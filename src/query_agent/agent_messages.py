from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool


def init_checkpointer(pool: ConnectionPool) -> PostgresSaver:
    """Build a ready-to-use PostgresSaver over `pool`: LangGraph's conversation-state checkpointer, keyed by thread_id.

    Runs the checkpointer's own (idempotent) schema setup before returning it.
    """
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer
