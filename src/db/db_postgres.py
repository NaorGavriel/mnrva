import os

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool, ConnectionPool

def _conninfo_from_env() -> str:
    """Build a conninfo string from POSTGRES_USER/PASSWORD/HOST/DB."""
    return make_conninfo(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )


def init_pool(conninfo: str | None = None) -> ConnectionPool:
    """Open a psycopg connection pool. `conninfo` defaults to one built from
    POSTGRES_USER/PORT/PASSWORD/HOST/DB."""
    return ConnectionPool(conninfo or _conninfo_from_env(), open=True, kwargs={"autocommit": True})


async def init_async_pool(conninfo: str | None = None) -> AsyncConnectionPool:
    """Async twin of init_pool."""
    pool = AsyncConnectionPool(conninfo or _conninfo_from_env(), open=False, kwargs={"autocommit": True})
    await pool.open()
    return pool


def ensure_repo_metadata_table(pool: ConnectionPool) -> None:
    """Create the single-row `repo_metadata` table if it doesn't already exist."""
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS repo_metadata (
                id INTEGER PRIMARY KEY DEFAULT 1,
                github_url TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT repo_metadata_single_row CHECK (id = 1)
            )
            """
        )


def ensure_checkpointer_tables(pool: ConnectionPool) -> None:
    """Create LangGraph's checkpointer tables if they don't already exist. Idempotent."""
    PostgresSaver(conn=pool).setup()


async def aensure_repo_metadata_table(pool: AsyncConnectionPool) -> None:
    """Async twin of ensure_repo_metadata_table."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS repo_metadata (
                id INTEGER PRIMARY KEY DEFAULT 1,
                github_url TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT repo_metadata_single_row CHECK (id = 1)
            )
            """
        )


async def aensure_checkpointer_tables(pool: AsyncConnectionPool) -> None:
    """Async twin of ensure_checkpointer_tables, for the query agent's startup. Idempotent."""
    await AsyncPostgresSaver(conn=pool).setup()
