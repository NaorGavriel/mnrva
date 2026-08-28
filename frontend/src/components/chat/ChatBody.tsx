import type { Message } from "../../hooks/useConversation.ts";
import UserMessage from "./UserMessage.tsx";
import AgentMessage from "./AgentMessage.tsx";
import "./ChatBody.css";

interface ChatBodyProps {
  messages: Message[];
}

/** The scrollable turn list. Messages are append-only within one conversation, so index is a stable key. */
export default function ChatBody({ messages }: ChatBodyProps) {
  return (
    <div className="chat-turns">
      {messages.map((message, index) =>
        message.role === "user" ? (
          <UserMessage key={index} text={message.text} />
        ) : (
          <AgentMessage key={index} turn={message} />
        ),
      )}
    </div>
  );
}
