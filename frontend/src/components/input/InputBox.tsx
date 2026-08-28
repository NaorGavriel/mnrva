import { useState, type SubmitEvent } from "react";
import { SendIcon } from "../../icons.tsx";
import type { Effort } from "../../types.ts";
import EffortSelector from "./EffortSelector.tsx";
import "./InputBox.css";

interface InputBoxProps {
  onSend: (text: string) => void;
  disabled: boolean;
  effort: Effort;
  onEffortChange: (effort: Effort) => void;
}

/** Text entry plus effort selector and send button. Owns the draft text locally - it's never part of conversation state until sent. */
export default function InputBox({ onSend, disabled, effort, onEffortChange }: InputBoxProps) {
  const [text, setText] = useState("");

  const handleSubmit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    const question = text.trim();
    if (!question || disabled) return;
    setText("");
    onSend(question);
  };

  return (
    <div className="chat-composer">
      <form className="chat-composer-card" onSubmit={handleSubmit}>
        <input
          className="chat-composer-input"
          type="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Ask about the codebase..."
          disabled={disabled}
        />
        <div className="chat-composer-toolbar">
          <EffortSelector value={effort} onChange={onEffortChange} disabled={disabled} />
          <button type="submit" className="chat-send" aria-label="Send" disabled={disabled || text.trim() === ""}>
            <SendIcon />
          </button>
        </div>
      </form>
    </div>
  );
}
