/** One parsed SSE frame: event name plus its raw (still-JSON-encoded) data. */
export interface SseFrame {
  event: string;
  data: string;
}

/** Splits one `event:`/`data:` frame (no trailing blank line) into its event name and joined data lines. Returns null for a frame with no data line. */
function parseSseFrame(frame: string): SseFrame | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }
  return dataLines.length > 0 ? { event, data: dataLines.join("\n") } : null;
}

/** Reads `response`'s body as a stream of SSE frames, buffering across chunk boundaries. Knows nothing about the frames' payload shape - just event name plus raw data. */
export async function* readSseFrames(response: Response): AsyncGenerator<SseFrame> {
  if (!response.body) {
    throw new Error("Response has no body to stream");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = parseSseFrame(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      if (frame) yield frame;
      boundary = buffer.indexOf("\n\n");
    }

    if (done) break;
  }
}
