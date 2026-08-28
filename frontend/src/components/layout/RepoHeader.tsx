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
      <h1 className="chat-repo-name">{repo ? repoNameFromUrl(repo.github_url) : "mnrva"}</h1>
      <p className="chat-tagline">indexed by mnrva</p>
      {repo && <p className="chat-synced">last synced {timeAgo(repo.updated_at)}</p>}
    </div>
  );
}
