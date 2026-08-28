import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { usePrefersReducedMotion } from "../../hooks/usePrefersReducedMotion.ts";
import "./AnswerText.css";

interface AnswerTextProps {
  text: string;
}

const WORD_INTERVAL_MS = 20;

/** Renders a finished answer with a simulated word-by-word typewriter reveal (the backend delivers the full answer in one shot). */
export default function AnswerText({ text }: AnswerTextProps) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const [revealed, setRevealed] = useState(prefersReducedMotion ? text : "");

  useEffect(() => {
    if (prefersReducedMotion) {
      setRevealed(text);
      return;
    }

    const words = text.split(/(\s+)/);
    let count = 0;
    setRevealed("");

    const id = setInterval(() => {
      count += 1;
      setRevealed(words.slice(0, count).join(""));
      if (count >= words.length) clearInterval(id);
    }, WORD_INTERVAL_MS);

    return () => clearInterval(id);
  }, [text, prefersReducedMotion]);

  return (
    <div className="chat-answer-text">
      <ReactMarkdown>{revealed}</ReactMarkdown>
    </div>
  );
}
