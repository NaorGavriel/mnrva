import "./ProgressStatus.css";

interface ProgressStatusProps {
  label: string;
}

/** Streaming-turn indicator: a pulsing dot plus the current node's step label. */
export default function ProgressStatus({ label }: ProgressStatusProps) {
  return (
    <div className="chat-status">
      <span className="chat-status-dot" />
      {label}
    </div>
  );
}
