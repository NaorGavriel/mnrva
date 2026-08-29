import { LogoMark } from "../../icons.tsx";
import NewConversationButton from "../input/NewConversationButton.tsx";
import "./Navbar.css";

interface NavbarProps {
  onNewConversation: () => void;
  disabled: boolean;
}

/** Fixed top bar: brand mark (also resets, like "click logo to go home") and the explicit reset button. */
export default function Navbar({ onNewConversation, disabled }: NavbarProps) {
  return (
    <header className="chat-topbar">
      <button type="button" className="chat-brand" onClick={onNewConversation} disabled={disabled}>
        <LogoMark />
        <span>mnrva</span>
      </button>
      <NewConversationButton onClick={onNewConversation} disabled={disabled} />
    </header>
  );
}
