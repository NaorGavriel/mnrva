/** Derives a short display name from a repo's github_url, e.g. "https://github.com/owner/repo" -> "repo". */
export function repoNameFromUrl(githubUrl: string): string {
  const segments = githubUrl.replace(/\/+$/, "").split("/");
  const last = segments[segments.length - 1] ?? githubUrl;
  return last.replace(/\.git$/i, "");
}
