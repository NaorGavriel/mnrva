import { useConversation } from "./hooks/useConversation.ts";
import { useRepoMetadata } from "./hooks/useRepoMetadata.ts";
import Navbar from "./components/layout/Navbar.tsx";
import RepoHeader from "./components/layout/RepoHeader.tsx";
import ChatBody from "./components/chat/ChatBody.tsx";
import InputBox from "./components/input/InputBox.tsx";
import ExamplePrompts from "./components/input/ExamplePrompts.tsx";
import "./App.css";

export default function App() {
  const { messages, status, effort, setEffort, sendMessage, newConversation } = useConversation();
  const { repo } = useRepoMetadata();
  const isLanding = messages.length === 0;

  return (
    <div className="chat">
      <Navbar onNewConversation={newConversation} disabled={isLanding} />
      <RepoHeader repo={repo} isLanding={isLanding} />

      {isLanding ? (
        <div className="chat-landing">
          <div className="chat-landing-composer">
            <InputBox onSend={sendMessage} disabled={status === "streaming"} effort={effort} onEffortChange={setEffort} />
            <ExamplePrompts onSelect={sendMessage} />
          </div>
        </div>
      ) : (
        <>
          <ChatBody messages={messages} />
          <div className="chat-composer-bar">
            <InputBox onSend={sendMessage} disabled={status === "streaming"} effort={effort} onEffortChange={setEffort} />
          </div>
        </>
      )}
    </div>
  );
}
