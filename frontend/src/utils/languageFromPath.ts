const EXTENSION_LANGUAGE: Record<string, string> = {
  py: "python",
  ts: "typescript",
  tsx: "tsx",
  js: "javascript",
  jsx: "jsx",
  md: "markdown",
  json: "json",
};

/** Maps a file path's extension to a react-syntax-highlighter (Prism) language name, per architecture.md's LANGUAGE_CONFIG. Falls back to "text" for unrecognized extensions. */
export function languageFromPath(filePath: string): string {
  const extension = filePath.split(".").pop()?.toLowerCase();
  return (extension && EXTENSION_LANGUAGE[extension]) ?? "text";
}

const MARKDOWN_EXTENSIONS = new Set(["md", "markdown"]);

/** Whether `filePath` is a markdown file - the only prose type that needs markdown rendering rather than a syntax-highlighted code block. */
export function isMarkdownPath(filePath: string): boolean {
  const extension = filePath.split(".").pop()?.toLowerCase();
  return extension !== undefined && MARKDOWN_EXTENSIONS.has(extension);
}
