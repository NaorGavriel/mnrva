import type { QueryStepNode } from "../../types.ts";
import "./ProgressStatus.css";

const STEP_ORDER: QueryStepNode[] = [
  "build_conversation_window",
  "evaluate_question",
  "retrieve_documents",
  "grade_documents",
  "generate_answer",
  "evaluate_answer",
];

const STEP_LABELS: Record<QueryStepNode, string> = {
  build_conversation_window: "Preparing conversation context",
  evaluate_question: "Understanding the question",
  retrieve_documents: "Searching the codebase",
  grade_documents: "Grading retrieved results",
  generate_answer: "Drafting an answer",
  evaluate_answer: "Checking answer quality",
};

interface ProgressStatusProps {
  currentStep?: QueryStepNode;
}

/** Streaming-turn indicator: a fixed 6-step breadcrumb plus the current step's label. The
 * corrective re-retrieval loop can revisit an earlier step - that just moves the current marker
 * back and re-dims the steps after it, without growing the trail past 6 slots. */
export default function ProgressStatus({ currentStep }: ProgressStatusProps) {
  const currentIndex = currentStep ? STEP_ORDER.indexOf(currentStep) : -1;

  return (
    <div className="chat-status">
      <div className="chat-status-steps">
        {STEP_ORDER.map((step, index) => (
          <span
            key={step}
            className={`chat-status-step${
              index < currentIndex ? " chat-status-step--done" : index === currentIndex ? " chat-status-step--current" : ""
            }`}
          />
        ))}
      </div>
      {currentStep ? STEP_LABELS[currentStep] : "Working..."}
    </div>
  );
}
