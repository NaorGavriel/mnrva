import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion.ts";
import "./AnswerText.css";

interface AnswerTextProps {
  text: string;
  onRevealComplete?: () => void;
}

const WORD_INTERVAL_MS = 20;

/** Renders a finished answer with a simulated word-by-word typewriter reveal (the backend delivers the full answer in one shot). Calls `onRevealComplete` once the last word lands (or immediately, under reduced motion). */
export default function AnswerText({ text, onRevealComplete }: AnswerTextProps) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const [revealed, setRevealed] = useState(prefersReducedMotion ? text : "");

  useEffect(() => {
    if (prefersReducedMotion) {
      setRevealed(text);
      onRevealComplete?.();
      return;
    }

    const words = text.split(/(\s+)/);
    let count = 0;
    setRevealed("");

    const id = setInterval(() => {
      count += 1;
      setRevealed(words.slice(0, count).join(""));
      if (count >= words.length) {
        clearInterval(id);
        onRevealComplete?.();
      }
    }, WORD_INTERVAL_MS);

    return () => clearInterval(id);
  }, [text, prefersReducedMotion, onRevealComplete]);

  return (
    <div className="chat-answer-text">
      <ReactMarkdown>{revealed}</ReactMarkdown>
    </div>
  );
}
