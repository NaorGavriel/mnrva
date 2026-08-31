# Install uv
FROM python:3.13-slim

# Installing git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.4 /uv /uvx /bin/

# Copy the project into the image
COPY . /app 
WORKDIR /app

# Sync the project
RUN uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "query_agent.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
