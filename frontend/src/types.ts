/** Mirrors query_agent/schemas.py and query_agent/api.py's request/SSE payload shapes. */

export type Effort = "basic" | "medium" | "high";

export interface QueryRequest {
  question: string;
  effort: Effort;
}

/** Mirrors schemas.Citation. start_line/end_line are null for prose-file citations. */
export interface Citation {
  chunk_id: string;
  file_path: string;
  start_line: number | null;
  end_line: number | null;
  citation_text: string;
}

/** Mirrors repo_metadata.RepoMetadata; updated_at arrives as an ISO 8601 string over JSON. */
export interface RepoMetadata {
  github_url: string;
  commit_sha: string;
  updated_at: string;
}

export interface ThreadResponse {
  thread_id: string;
}

/** graph.py node names that stream a plain progress update (query_agent/api.py's _STEP_LABELS, minus persist_agent_message). */
export type QueryStepNode =
  | "build_conversation_window"
  | "evaluate_question"
  | "retrieve_documents"
  | "grade_documents"
  | "generate_answer"
  | "evaluate_answer";

export interface QueryStepUpdate {
  label: string;
}

/** persist_agent_message's payload: the terminal update, carrying the finished turn. */
export interface QueryFinalUpdate {
  label: string;
  answer: string;
  citations: Citation[];
}

/** Emitted if the turn fails mid-stream, in place of persist_agent_message. */
export interface QueryErrorUpdate {
  message: string;
}

export type QueryTurnEvent =
  | { node: QueryStepNode; data: QueryStepUpdate }
  | { node: "persist_agent_message"; data: QueryFinalUpdate }
  | { node: "error"; data: QueryErrorUpdate };
