import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import ReactMarkdown from "react-markdown";
import { FileIcon } from "../../icons.tsx";
import type { Citation as CitationType } from "../../types.ts";
import { isMarkdownPath, languageFromPath } from "../../utils/languageFromPath.ts";
import { usePrefersDark } from "../../hooks/usePrefersDark.ts";
import "./Citation.css";

interface CitationProps {
  citation: CitationType;
}

/** One collapsible, syntax-highlighted citation excerpt - code via Prism, prose (.md) via markdown. */
export default function Citation({ citation }: CitationProps) {
  const codeStyle = usePrefersDark() ? oneDark : oneLight;

  return (
    <details className="citation">
      <summary>
        <span className="citation-chevron" aria-hidden="true" />
        <FileIcon className="citation-file-icon" />
        <span className="citation-path">{citation.file_path}</span>
        {citation.start_line !== null && citation.end_line !== null && (
          <span className="citation-lines">
            L{citation.start_line}-{citation.end_line}
          </span>
        )}
      </summary>
      <div className="citation-body">
        {isMarkdownPath(citation.file_path) ? (
          <div className="citation-prose">
            <ReactMarkdown>{citation.citation_text}</ReactMarkdown>
          </div>
        ) : (
          <SyntaxHighlighter
            language={languageFromPath(citation.file_path)}
            style={codeStyle}
            customStyle={{
              margin: 0,
              padding: "12px 14px",
              background: "transparent",
              fontSize: "13px",
              whiteSpace: "pre",
              overflowX: "auto",
            }}
          >
            {citation.citation_text}
          </SyntaxHighlighter>
        )}
      </div>
    </details>
  );
}
