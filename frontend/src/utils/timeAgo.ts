const UNITS: [Intl.RelativeTimeFormatUnit, number][] = [
  ["year", 60 * 60 * 24 * 365],
  ["month", 60 * 60 * 24 * 30],
  ["week", 60 * 60 * 24 * 7],
  ["day", 60 * 60 * 24],
  ["hour", 60 * 60],
  ["minute", 60],
  ["second", 1],
];

const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "always" });

/** Formats an ISO timestamp as a relative "X time ago" string. */
export function timeAgo(isoTimestamp: string): string {
  const elapsedSeconds = (new Date(isoTimestamp).getTime() - Date.now()) / 1000;
  const absSeconds = Math.abs(elapsedSeconds);

  for (const [unit, secondsInUnit] of UNITS) {
    if (absSeconds >= secondsInUnit) {
      return formatter.format(Math.round(elapsedSeconds / secondsInUnit), unit);
    }
  }
  return formatter.format(Math.round(elapsedSeconds), "second");
}
