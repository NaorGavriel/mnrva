import os

from psycopg.conninfo import make_conninfo
from psycopg_pool import ConnectionPool

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
    return ConnectionPool(conninfo or _conninfo_from_env(), open=True)


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
