import { useEffect, useRef, useState } from "react";

import { runFramesWsUrl } from "@/lib/api";

export interface LiveClip {
  iter: number;
  url: string;
  durationS?: number;
  sizeBytes?: number;
}

export interface SkippedClip {
  iter: number;
  reason: string;
}

export interface UseLiveClipsState {
  clips: LiveClip[];
  skipped: SkippedClip[];
  connected: boolean;
  terminal: boolean;
}

/** Subscribe to the `/frames` WS. Clips arrive via the backend's
 *  RolloutStreamer — typically one per iteration, ~2s each. On
 *  connect the WS replays every prior clip event so the replay strip
 *  is fully populated without an extra REST call. */
export function useLiveClips(
  slug: string | undefined,
  runId: string | undefined,
): UseLiveClipsState {
  const [clips, setClips] = useState<LiveClip[]>([]);
  const [skipped, setSkipped] = useState<SkippedClip[]>([]);
  const [connected, setConnected] = useState(false);
  const [terminal, setTerminal] = useState(false);
  // §Ship 21e (review fix, CRITICAL): mirror `terminal` into a ref so
  // `onclose` reads the CURRENT value, not the stale closure captured
  // at WS-registration time. Without this, completing a run while
  // Live mode is open triggers an INFINITE reconnect loop: server
  // closes the WS when the job ends → onclose sees terminal=false
  // (the closure value) → schedules a reconnect → server replays the
  // terminal event → onclose sees stale false again → reconnects
  // forever. This is the exact bug useRunEvents/useJobEvents already
  // guard against (terminalRef); useLiveClips was missed.
  const terminalRef = useRef(false);

  useEffect(() => {
    setClips([]);
    setSkipped([]);
    setConnected(false);
    setTerminal(false);
    terminalRef.current = false;
    if (!slug || !runId) return;

    let cancelled = false;
    let ws: WebSocket | null = null;
    let retry = 0;
    // §Ship 21e: capture the reconnect timer so cleanup can clear it
    // (prevents a stray fire after unmount).
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled || terminalRef.current) return;
      ws = new WebSocket(runFramesWsUrl(slug, runId));

      ws.onopen = () => {
        retry = 0;
        setConnected(true);
      };

      ws.onmessage = (msg) => {
        let ev: { type?: string; iter?: number; url?: string; duration_s?: number; size_bytes?: number; reason?: string };
        try {
          ev = JSON.parse(msg.data);
        } catch {
          return;
        }
        if (ev.type === "new_clip" && typeof ev.iter === "number" && typeof ev.url === "string") {
          const clip: LiveClip = {
            iter: ev.iter,
            url: ev.url,
            durationS: typeof ev.duration_s === "number" ? ev.duration_s : undefined,
            sizeBytes: typeof ev.size_bytes === "number" ? ev.size_bytes : undefined,
          };
          setClips((xs) => mergeClip(xs, clip));
        } else if (ev.type === "clip_skipped" && typeof ev.iter === "number") {
          setSkipped((xs) => [...xs, { iter: ev.iter!, reason: ev.reason ?? "unknown" }]);
        } else if (ev.type === "terminal") {
          terminalRef.current = true;
          setTerminal(true);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (cancelled || terminalRef.current) return;
        const delay = Math.min(8000, 250 * Math.pow(2, retry));
        retry++;
        reconnectTimer = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        // handled by onclose
      };
    };

    connect();
    return () => {
      cancelled = true;
      if (reconnectTimer !== null) clearTimeout(reconnectTimer);
      try {
        ws?.close();
      } catch {
        // ignore
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slug, runId]);

  return { clips, skipped, connected, terminal };
}

function mergeClip(existing: LiveClip[], next: LiveClip): LiveClip[] {
  // Replace if we already saw this iter (the server re-emits on
  // reconnect); otherwise append in sorted-by-iter order.
  const idx = existing.findIndex((c) => c.iter === next.iter);
  if (idx >= 0) {
    const out = existing.slice();
    out[idx] = next;
    return out;
  }
  const out = [...existing, next];
  out.sort((a, b) => a.iter - b.iter);
  return out;
}
