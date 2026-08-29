import type { Citation as CitationType } from "../../types.ts";
import Citation from "./Citation.tsx";
import "./CitationList.css";

interface CitationListProps {
  citations: CitationType[];
}

/** Lays out one answer's backing citations. Renders nothing when there are none. */
export default function CitationList({ citations }: CitationListProps) {
  if (citations.length === 0) return null;

  return (
    <div className="citation-list">
      {citations.map((citation, index) => (
        <Citation key={citation.chunk_id} citation={citation} revealDelayMs={index * 60} />
      ))}
    </div>
  );
}
