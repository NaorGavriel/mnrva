import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from repository_ingester import ingest_repository


def main() -> None:
    """Entrypoint for the `mnrva-ingest` console script: ingest the given GitHub repo URL, exit non-zero on failure."""
    if len(sys.argv) != 2:
        print("usage: mnrva-ingest <github_url>")
        sys.exit(1)
    github_url = sys.argv[1]

    try:
        commit_sha = asyncio.run(ingest_repository(github_url))
    except Exception as exc:
        print(f"ingest: failed - {exc}")
        sys.exit(1)

    print(f"ingest: {github_url} @ {commit_sha}")
