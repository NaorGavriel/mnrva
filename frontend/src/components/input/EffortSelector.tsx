import type { Effort } from "../../types.ts";
import "./EffortSelector.css";

const EFFORT_OPTIONS: Effort[] = ["basic", "medium", "high"];
const EFFORT_LABELS: Record<Effort, string> = { basic: "Basic", medium: "Medium", high: "High" };

interface EffortSelectorProps {
  value: Effort;
  onChange: (effort: Effort) => void;
  disabled: boolean;
}

/** basic/medium/high toggle group, sent as QueryRequest.effort. */
export default function EffortSelector({ value, onChange, disabled }: EffortSelectorProps) {
  return (
    <div className="chat-effort" role="group" aria-label="Reasoning effort">
      {EFFORT_OPTIONS.map((option) => (
        <button
          key={option}
          type="button"
          className={option === value ? "is-active" : ""}
          onClick={() => onChange(option)}
          disabled={disabled}
        >
          {EFFORT_LABELS[option]}
        </button>
      ))}
    </div>
  );
}
