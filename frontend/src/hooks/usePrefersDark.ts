import { useEffect, useState } from "react";

/** Tracks the OS/browser's prefers-color-scheme, updating live if the user switches while the page is open. */
export function usePrefersDark(): boolean {
  const [prefersDark, setPrefersDark] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);

  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = (event: MediaQueryListEvent) => setPrefersDark(event.matches);
    query.addEventListener("change", listener);
    return () => query.removeEventListener("change", listener);
  }, []);

  return prefersDark;
}
