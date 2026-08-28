import "./ExamplePrompts.css";

const EXAMPLE_PROMPTS = [
  "Give me an overview of this codebase",
  "How is the project structured?",
  "Where would I start if I wanted to add a new feature?",
];

interface ExamplePromptsProps {
  onSelect: (text: string) => void;
}

/** Landing-page starter prompts, repo-agnostic since the frontend doesn't know which repo is loaded. Clicking one sends it immediately, bypassing InputBox's draft text. */
export default function ExamplePrompts({ onSelect }: ExamplePromptsProps) {
  return (
    <div className="example-prompts">
      {EXAMPLE_PROMPTS.map((prompt) => (
        <button key={prompt} type="button" className="example-prompt" onClick={() => onSelect(prompt)}>
          {prompt}
        </button>
      ))}
    </div>
  );
}
