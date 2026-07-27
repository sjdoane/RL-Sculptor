/** Compact RFC 3339 timestamp formatter — e.g. "2 min ago", "Apr 20".
 *  Used on ProjectCards where space is tight. */
export function formatRelative(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const diffMs = Date.now() - d.getTime();
  const mins = Math.round(diffMs / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.round(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** Strips ANSI escape sequences (SGR color codes, cursor movement, OSC
 *  title/hyperlink sequences) and other C0/C1 control characters from
 *  text that was captured off raw training-console output — diagnoser
 *  evidence, reward-edit reasoning — before it's rendered. Progress-bar
 *  style output (tqdm, rich) packs many ESC sequences into one line;
 *  left un-stripped, each renders as an unrenderable tofu/box glyph in
 *  most fonts, which is what shows up as rows of black boxes. Keeps
 *  `\n`/`\t` so `whiteSpace: "pre-wrap"` formatting still works. Pure —
 *  call at render time only, never mutate the stored/fetched data. */
export function sanitizeConsoleText(text: string): string {
  if (!text) return text;
  return text
    // CSI sequences: ESC [ params intermediates final-byte (colors, cursor moves, clear-line, ...)
    .replace(/\x1B\[[0-?]*[ -/]*[@-~]/g, "")
    // OSC sequences: ESC ] ... terminated by BEL or ESC \ (window title, hyperlinks, ...)
    .replace(/\x1B\][^\x07\x1B]*(?:\x07|\x1B\\)/g, "")
    // Other two-byte Fe escapes (cursor save/restore, charset select, ...)
    .replace(/\x1B[@-Z\\^_]/g, "")
    // Any leftover lone ESC that didn't match a known sequence shape
    .replace(/\x1B/g, "")
    // Remaining C0 (minus \t \n) and C1 control chars — no printable glyph
    .replace(/[\x00-\x08\x0B\x0C\x0D\x0E-\x1F\x7F-\x9F]/g, "");
}
