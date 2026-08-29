import { useCallback, useState } from "react";
import { LogoMark } from "../../icons.tsx";
import type { AgentTurn } from "../../hooks/useConversation.ts";
import AnswerText from "./AnswerText.tsx";
import CitationList from "../citations/CitationList.tsx";
import ProgressStatus from "./ProgressStatus.tsx";
import ErrorMessage from "./ErrorMessage.tsx";
import "./AgentMessage.css";

interface AgentMessageProps {
  turn: AgentTurn;
}

/** One agent turn: branches on `phase` to show streaming progress, a terminal error, or the
 * finished answer plus citations. Citations only mount once the answer's reveal finishes */
export default function AgentMessage({ turn }: AgentMessageProps) {
  const [citationsVisible, setCitationsVisible] = useState(false);
  const handleRevealComplete = useCallback(() => setCitationsVisible(true), []);

  return (
    <div className="chat-answer">
      <div className="chat-answer-label">
        <LogoMark className="chat-answer-icon" />
        mnrva
      </div>
      {turn.phase === "streaming" && <ProgressStatus currentStep={turn.currentStep} />}
      {turn.phase === "error" && <ErrorMessage message={turn.error ?? "Something went wrong."} />}
      {turn.phase === "done" && (
        <>
          <AnswerText text={turn.answer ?? ""} onRevealComplete={handleRevealComplete} />
          {citationsVisible && <CitationList citations={turn.citations ?? []} />}
        </>
      )}
    </div>
  );
}
