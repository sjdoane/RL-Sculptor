import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { runEventsWsUrl } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { RunEvent } from "@/lib/types";

/** Subscribe to a run's live event stream.
 *
 *  The WS endpoint replays the last 200 log lines + all typed events
 *  on connect, then streams new events. On disconnect we reconnect
 *  with exponential backoff up to ~8s, passing `seq` to skip events
 *  the client already saw (the server replays from its full `events`
 *  list anyway; we dedupe client-side).
 */
export interface UseRunEventsState {
  events: RunEvent[];
  connected: boolean;
  terminal: boolean;
}

export function useRunEvents(
  slug: string | undefined,
  runId: string | undefined,
): UseRunEventsState {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [terminal, setTerminal] = useState(false);
  const seqRef = useRef<number>(0);
  // Mirror `terminal` into a ref so the `onclose` handler below reads
  // the CURRENT value instead of the stale closure value captured at
  // WS-registration time. Without this, killing a run while the tab
  // is open triggers an infinite reconnect loop: server closes the
  // WS when the job ends, onclose still sees `terminal=false`, so it
  // schedules a reconnect, which replays the last N events, which
  // triggers another close, etc. (visible as repeated
  // "connected (replay_count=N)" log lines in the UI + server).
  const terminalRef = useRef(false);
  const qc = useQueryClient();

  useEffect(() => {
    setEvents([]);
    seqRef.current = 0;
    setConnected(false);
    setTerminal(false);
    terminalRef.current = false;
    if (!slug || !runId) return;

    let cancelled = false;
    let ws: WebSocket | null = null;
    let retry = 0;

    const connect = () => {
      if (cancelled || terminalRef.current) return;
      const url = runEventsWsUrl(slug, runId);
      ws = new WebSocket(url);

      ws.onopen = () => {
        retry = 0;
        setConnected(true);
      };

      ws.onmessage = (msg) => {
        let ev: RunEvent;
        try {
          ev = JSON.parse(msg.data);
        } catch {
          return;
        }
        if (ev.type === "terminal") {
          terminalRef.current = true;
          setTerminal(true);
          return;
        }
        if (typeof ev.seq === "number") {
          if (ev.seq <= seqRef.current) return; // dedupe replay
          seqRef.current = ev.seq;
        }
        setEvents((xs) => {
          // Cap at 20k entries in memory to bound memory on very long
          // runs — the server's log file is still the source of truth.
          const next = [...xs, ev];
          if (next.length > 20_000) next.splice(0, next.length - 20_000);
          return next;
        });
        if (ev.type === "run_completed" || ev.type === "run_errored" || ev.type === "run_stopped") {
          // Invalidate REST queries so the detail view refreshes with
          // the final summary.
          if (slug && runId) {
            qc.invalidateQueries({ queryKey: qk.run(slug, runId) });
            qc.invalidateQueries({ queryKey: qk.runs(slug) });
          }
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (cancelled || terminalRef.current) return;
        const delay = Math.min(8000, 250 * Math.pow(2, retry));
        retry++;
        setTimeout(connect, delay);
      };

      ws.onerror = () => {
        // Handled by onclose — nothing extra to do here.
      };
    };

    connect();

    return () => {
      cancelled = true;
      try {
        ws?.close();
      } catch {
        // ignore
      }
    };
  }, [slug, runId, qc]);

  return { events, connected, terminal };
}
