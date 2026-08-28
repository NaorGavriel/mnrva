import { useEffect, useState } from "react";
import { getRepoMetadata } from "../api/client.ts";
import type { RepoMetadata } from "../types.ts";

interface UseRepoMetadata {
  repo: RepoMetadata | null;
  loading: boolean;
}

/** Fetches the tracked repo's metadata once on mount. `repo` stays null if the fetch fails (e.g. nothing ingested yet) - callers fall back to a placeholder rather than showing an error for this. */
export function useRepoMetadata(): UseRepoMetadata {
  const [repo, setRepo] = useState<RepoMetadata | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    getRepoMetadata(controller.signal)
      .then(setRepo)
      .catch(() => {})
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  return { repo, loading };
}
