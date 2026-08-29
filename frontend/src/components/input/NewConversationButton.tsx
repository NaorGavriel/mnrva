import { PlusIcon } from "../../icons.tsx";
import "./NewConversationButton.css";

interface NewConversationButtonProps {
  onClick: () => void;
  disabled: boolean;
}

/** Resets the conversation: clears messages and thread_id, without a full page reload. */
export default function NewConversationButton({ onClick, disabled }: NewConversationButtonProps) {
  return (
    <button type="button" className="chat-new-btn" onClick={onClick} disabled={disabled}>
      <PlusIcon />
      <span className="chat-new-btn-label">New conversation</span>
    </button>
  );
}
