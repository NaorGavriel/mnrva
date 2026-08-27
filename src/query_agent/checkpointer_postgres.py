from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from db.db_postgres import aensure_checkpointer_tables, ensure_checkpointer_tables
from query_agent.schemas import Answer
from query_agent.effort import BasicEffort, MediumEffort, HighEffort
from psycopg_pool import AsyncConnectionPool, ConnectionPool

_ALLOWED_MSGPACK_MODULES = [BasicEffort, Answer, MediumEffort, HighEffort]


def init_checkpointer(pool: ConnectionPool) -> PostgresSaver:
    """Build a ready-to-use PostgresSaver over `pool`: LangGraph's conversation-state checkpointer, keyed by thread_id.

    Runs the checkpointer's own (idempotent) schema setup before returning it.
    """
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    checkpointer = PostgresSaver(conn=pool, serde=serde)
    ensure_checkpointer_tables(pool)
    return checkpointer


async def init_async_checkpointer(pool: AsyncConnectionPool) -> AsyncPostgresSaver:
    """Async twin of init_checkpointer, over an async pool."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    checkpointer = AsyncPostgresSaver(conn=pool, serde=serde)
    await aensure_checkpointer_tables(pool)
    return checkpointer
