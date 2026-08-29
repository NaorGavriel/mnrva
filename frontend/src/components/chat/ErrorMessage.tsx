import "./ErrorMessage.css";

interface ErrorMessageProps {
  message: string;
}

/** Inline error state for a turn that failed mid-stream (the SSE `error` event). */
export default function ErrorMessage({ message }: ErrorMessageProps) {
  return <div className="chat-error">{message}</div>;
}
