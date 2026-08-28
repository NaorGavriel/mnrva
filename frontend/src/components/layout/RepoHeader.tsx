import type { RepoMetadata } from "../../types.ts";
import { repoNameFromUrl } from "../../utils/repoName.ts";
import { timeAgo } from "../../utils/timeAgo.ts";
import "./RepoHeader.css";

interface RepoHeaderProps {
  repo: RepoMetadata | null;
  isLanding: boolean;
}

/** Fixed, centered repo identity - github_url's short name and last-synced time - that animates from a large landing headline into a small top-center label once the conversation starts. */
export default function RepoHeader({ repo, isLanding }: RepoHeaderProps) {
  return (
    <div className={`chat-header-text${isLanding ? "" : " chat-header-text--chat"}`}>
      <div className="chat-title-group">
        <h1 className="chat-repo-name">{repo ? repoNameFromUrl(repo.github_url) : "mnrva"}</h1>
        <p className="chat-tagline">
          indexed by <span className="chat-tagline-accent">mnrva</span>
        </p>
      </div>
      {repo && <p className="chat-synced">last synced {timeAgo(repo.updated_at)}</p>}
    </div>
  );
}
