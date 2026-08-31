import type { QueryRequest, QueryTurnEvent, RepoMetadata, ThreadResponse } from "../types.ts";
import { readSseFrames } from "./sse.ts";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/** Creates a new conversation thread. Stateless server-side (query_agent/api.py just mints a uuid4). */
export async function createThread(signal?: AbortSignal): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/threads`, { method: "POST", signal });
  if (!response.ok) {
    throw new Error(`Failed to create thread: ${response.status} ${response.statusText}`);
  }
  const body: ThreadResponse = await response.json();
  return body.thread_id;
}

/** Fetches the tracked repo's github_url/commit_sha/updated_at. */
export async function getRepoMetadata(signal?: AbortSignal): Promise<RepoMetadata> {
  const response = await fetch(`${API_BASE_URL}/repo`, { signal });
  if (!response.ok) {
    throw new Error(`Failed to fetch repo metadata: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

/** Streams one query-agent turn's per-node progress, yielding one QueryTurnEvent per SSE frame. */
export async function* streamQuery(
  threadId: string,
  request: QueryRequest,
  signal?: AbortSignal,
): AsyncGenerator<QueryTurnEvent> {
  const response = await fetch(`${API_BASE_URL}/threads/${threadId}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok) {
    throw new Error(`Query request failed: ${response.status} ${response.statusText}`);
  }

  for await (const frame of readSseFrames(response)) {
    yield { node: frame.event, data: JSON.parse(frame.data) } as QueryTurnEvent;
  }
}
