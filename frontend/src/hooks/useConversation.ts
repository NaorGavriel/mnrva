import { useCallback, useEffect, useRef, useState } from "react";
import { createThread, streamQuery } from "../api/client.ts";
import type { Citation, Effort, QueryStepNode } from "../types.ts";

export interface UserTurn {
  role: "user";
  text: string;
}

export interface AgentTurn {
  role: "agent";
  phase: "streaming" | "done" | "error";
  currentStep?: QueryStepNode;
  answer?: string;
  citations?: Citation[];
  error?: string;
}

export type Message = UserTurn | AgentTurn;

interface UseConversation {
  messages: Message[];
  status: "idle" | "streaming";
  effort: Effort;
  setEffort: (effort: Effort) => void;
  sendMessage: (text: string) => void;
  newConversation: () => void;
}

/** Owns one conversation's turn history and streaming state. Lazily creates a thread_id on the
 * first message; `newConversation` resets everything - thread_id, messages, and effort - back to
 * a fresh start. */
export function useConversation(): UseConversation {
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<"idle" | "streaming">("idle");
  const [effort, setEffort] = useState<Effort>("basic");
  const threadId = useRef<string | null>(null);
  const abortController = useRef<AbortController | null>(null);

  const updateLastAgentTurn = useCallback((update: Partial<AgentTurn>) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role !== "agent") return prev;
      return [...prev.slice(0, -1), { ...last, ...update }];
    });
  }, []);

  const sendMessage = useCallback(
    (text: string) => {
      if (status === "streaming") return;

      setMessages((prev) => [...prev, { role: "user", text }, { role: "agent", phase: "streaming" }]);
      setStatus("streaming");

      const controller = new AbortController();
      abortController.current = controller;

      (async () => {
        try {
          if (!threadId.current) {
            threadId.current = await createThread(controller.signal);
          }
          for await (const event of streamQuery(threadId.current, { question: text, effort }, controller.signal)) {
            if (event.node === "persist_agent_message") {
              updateLastAgentTurn({ phase: "done", answer: event.data.answer, citations: event.data.citations });
            } else if (event.node === "error") {
              updateLastAgentTurn({ phase: "error", error: event.data.message });
            } else {
              updateLastAgentTurn({ currentStep: event.node });
            }
          }
        } catch (err) {
          if (controller.signal.aborted) return;
          updateLastAgentTurn({ phase: "error", error: err instanceof Error ? err.message : "Something went wrong" });
        } finally {
          if (!controller.signal.aborted) setStatus("idle");
        }
      })();
    },
    [status, effort, updateLastAgentTurn],
  );

  useEffect(() => {
    return () => abortController.current?.abort();
  }, []);

  const newConversation = useCallback(() => {
    abortController.current?.abort();
    threadId.current = null;
    setMessages([]);
    setEffort("basic");
    setStatus("idle");
  }, []);

  return { messages, status, effort, setEffort, sendMessage, newConversation };
}
