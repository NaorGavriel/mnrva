import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from db.db_postgres import init_pool
from db.db_qdrant import QDRANT_API_KEY, QDRANT_URL, init_client
from refresh_sync import sync_repository

REFRESH_FILES_DIR = Path("refresh_files")


def main() -> None:
    """Entrypoint for the `mnrva-refresh` console script: run one refresh cycle, exit non-zero on failure."""
    pool = init_pool()
    client = init_client(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    try:
        result = asyncio.run(sync_repository(pool, client, REFRESH_FILES_DIR))
    except Exception as exc:
        print(f"refresh: failed - {exc}")
        sys.exit(1)

    print(
        f"refresh: {result['old_sha']} -> {result['new_sha']} "
        f"(+{result['added']} ~{result['modified']} -{result['deleted']})"
    )
