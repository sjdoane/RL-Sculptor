import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { missionEventsWsUrl } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import type { MissionEvent } from "@/lib/types";

/** Subscribe to a mission's live event stream.
 *
 *  Per Ship 18a audit-deferred finding A + Ship 18b handoff §6: the
 *  caller MUST gate `enabled` on either (a) a runMission mutation
 *  having just succeeded, OR (b) the mission's `active_job_id` being
 *  set on the latest summary/detail. Pre-mounting the WS would attach
 *  to a no-active-job server response and close immediately, leaving
 *  the user with a confusing empty stream.
 *
 *  v1 does NOT auto-reconnect on transient disconnect (Ship 18b
 *  handoff §6 explicit decision). Instead, when onclose fires before
 *  the terminal event, `disconnected` is set and the UI should show a
 *  "WebSocket disconnected — refresh to retry" banner. Auto-reconnect
 *  with replay-from-seq is a Ship 18c follow-up.
 */
export interface UseMissionEventsState {
  logLines: MissionEvent[];
  structuredEvents: MissionEvent[];
  connected: boolean;
  terminal: boolean;
  disconnected: boolean;
  noActiveJob: boolean;
}

const LOG_LINE_CAP = 200;
const STRUCTURED_EVENT_CAP = 5000;

const TERMINAL_EVENT_TYPES = new Set<string>([
  "mission_decompose_completed",
  "mission_decompose_errored",
  "mission_execute_completed",
  "mission_execute_errored",
  "mission_completed",
  "mission_halted_terminal",
]);

const STAGE_OR_MISSION_REFRESH_PREFIXES = [
  "stage_",
  "mission_",
  "redecomposition_",
];

export function useMissionEvents(
  slug: string | undefined,
  missionSlug: string | undefined,
  enabled: boolean,
): UseMissionEventsState {
  const [logLines, setLogLines] = useState<MissionEvent[]>([]);
  const [structuredEvents, setStructuredEvents] = useState<MissionEvent[]>(
    [],
  );
  const [connected, setConnected] = useState(false);
  const [terminal, setTerminal] = useState(false);
  const [disconnected, setDisconnected] = useState(false);
  const [noActiveJob, setNoActiveJob] = useState(false);

  // Mirror terminal / no_active_job into refs so the `onclose` handler
  // reads the current value, not the stale closure capture from when
  // the WS was opened. Without this, `disconnected` would flip to true
  // immediately after a clean terminal close. Pattern from
  // useRunEvents.ts:38.
  const terminalRef = useRef(false);
  const noActiveJobRef = useRef(false);

  const qc = useQueryClient();

  useEffect(() => {
    setLogLines([]);
    setStructuredEvents([]);
    setConnected(false);
    setTerminal(false);
    setDisconnected(false);
    setNoActiveJob(false);
    terminalRef.current = false;
    noActiveJobRef.current = false;

    if (!enabled || !slug || !missionSlug) return;

    let cancelled = false;
    let ws: WebSocket | null = null;

    const url = missionEventsWsUrl(slug, missionSlug);
    ws = new WebSocket(url);

    ws.onopen = () => {
      if (!cancelled) setConnected(true);
    };

    ws.onmessage = (msg) => {
      // Audit fix (Ship 18b): if cleanup already ran, drop in-flight
      // messages so a re-subscription (slug/missionSlug/enabled change)
      // doesn't append events from the previous WS into the new state.
      if (cancelled) return;
      let ev: MissionEvent;
      try {
        ev = JSON.parse(msg.data);
      } catch {
        return;
      }

      const type = ev.type;

      if (type === "connected") {
        const status = (ev as { status?: string }).status;
        if (status === "no_active_job") {
          noActiveJobRef.current = true;
          setNoActiveJob(true);
        }
        return;
      }

      if (type === "terminal") {
        terminalRef.current = true;
        setTerminal(true);
        if (slug && missionSlug) {
          qc.invalidateQueries({
            queryKey: qk.mission(slug, missionSlug),
          });
          qc.invalidateQueries({ queryKey: qk.missions(slug) });
        }
        return;
      }

      if (type === "log_line") {
        setLogLines((xs) => {
          const next = xs.length >= LOG_LINE_CAP ? xs.slice(1) : xs.slice();
          next.push(ev);
          return next;
        });
        return;
      }

      // Structured event. Cap at STRUCTURED_EVENT_CAP so a long-running
      // mission with many re-decompositions doesn't grow without bound
      // (plan-audit CRITICAL #3).
      setStructuredEvents((xs) => {
        const next =
          xs.length >= STRUCTURED_EVENT_CAP ? xs.slice(1) : xs.slice();
        next.push(ev);
        return next;
      });

      // Stage / mission lifecycle events change on-disk state; nudge
      // React Query to refetch the detail so the stage list reflects
      // the new status without waiting for terminal.
      if (
        slug &&
        missionSlug &&
        STAGE_OR_MISSION_REFRESH_PREFIXES.some((p) => type.startsWith(p)) &&
        TERMINAL_EVENT_TYPES.has(type) === false
      ) {
        qc.invalidateQueries({
          queryKey: qk.mission(slug, missionSlug),
        });
      }
    };

    ws.onclose = () => {
      if (cancelled) return;
      setConnected(false);
      if (terminalRef.current || noActiveJobRef.current) return;
      // No reconnect (Ship 18b handoff §6); surface a banner instead.
      setDisconnected(true);
    };

    ws.onerror = () => {
      // Handled by onclose.
    };

    return () => {
      cancelled = true;
      try {
        ws?.close();
      } catch {
        // ignore
      }
    };
  }, [slug, missionSlug, enabled, qc]);

  return {
    logLines,
    structuredEvents,
    connected,
    terminal,
    disconnected,
    noActiveJob,
  };
}
