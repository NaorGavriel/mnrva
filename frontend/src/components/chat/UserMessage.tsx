import "./UserMessage.css";

interface UserMessageProps {
  text: string;
}

/** One user turn, right-aligned. */
export default function UserMessage({ text }: UserMessageProps) {
  return (
    <div className="chat-question">
      <span className="chat-bubble">{text}</span>
    </div>
  );
}
