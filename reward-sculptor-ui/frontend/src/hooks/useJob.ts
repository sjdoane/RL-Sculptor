import { useEffect, useRef, useState } from "react";
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { getJob, jobEventsWsUrl, stopJob } from "@/lib/api";
import { qk } from "@/lib/queryKeys";
import {
  JOB_TERMINAL_STATES,
  type JobDetail,
  type JobStatus,
  type JobSummary,
} from "@/lib/types";

/** One log_line / typed event streamed from /ws/jobs/{id}/events. */
export interface JobEvent {
  type: string;
  seq?: number;
  ts?: string;
  text?: string;
  status?: string;
  [key: string]: unknown;
}

export interface UseJobEventsState {
  events: JobEvent[];
  connected: boolean;
  terminal: boolean;
}


export interface UseJobOptions {
  /** Polling interval while the job is active. Default 3000 ms per
   *  Prompt 7 precondition R3 — not 2000 ms. */
  intervalMs?: number;
  /** Set to `false` once the job handle is consumed so we don't
   *  keep polling dead jobs. */
  enabled?: boolean;
  /** Fired exactly once when status transitions to any terminal state
   *  (completed / errored / stopped). */
  onTerminal?: (job: JobDetail) => void;
}

/** Poll a single job until it reaches a terminal state. The hook
 *  invalidates related queries on completion (papers, techniques,
 *  pending seeds) so dependent tabs refetch without extra wiring. */
export function useJob(
  jobId: string | undefined,
  options: UseJobOptions = {},
) {
  const { intervalMs = 3000, enabled = true, onTerminal } = options;
  const qc = useQueryClient();

  const query = useQuery<JobDetail>({
    queryKey: jobId ? qk.job(jobId) : ["job", "_none"],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId && enabled,
    refetchInterval: (q) => {
      const data = q.state.data as JobDetail | undefined;
      if (!data) return intervalMs;
      if (JOB_TERMINAL_STATES.includes(data.status)) return false;
      return intervalMs;
    },
    staleTime: 1000,
  });

  const status = query.data?.status;
  useEffect(() => {
    if (!query.data) return;
    if (!status) return;
    if (!JOB_TERMINAL_STATES.includes(status)) return;
    // Invalidate everything downstream of the job.
    const slug = query.data.project_slug;
    if (slug) {
      qc.invalidateQueries({ queryKey: ["kg", "papers", slug] });
      qc.invalidateQueries({ queryKey: qk.kgPendingSeeds(slug) });
      qc.invalidateQueries({ queryKey: qk.kgTechniques(slug) });
    }
    if (onTerminal) onTerminal(query.data);
    // We intentionally run this effect only when status flips — not
    // on every data change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  return query;
}

export function isTerminal(status: JobStatus): boolean {
  return JOB_TERMINAL_STATES.includes(status);
}

/** Subscribe to a single job's event stream (reward/physics prompt
 *  edit, KG extract). Replays buffered log_lines on connect, tails
 *  live emits, closes on terminal. Mirrors `useRunEvents` but keyed
 *  on job_id. Used by RewardsTab's Activity panel (bug #6.1 / T2).
 *
 *  Pass `undefined` to disable — e.g. when no job is in flight. */
export function useJobEvents(
  jobId: string | undefined,
): UseJobEventsState {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [terminal, setTerminal] = useState(false);
  const seqRef = useRef<number>(0);
  // Mirror `terminal` into a ref so onclose sees the CURRENT value,
  // not the stale closure captured at WS-registration time — same
  // pattern as useRunEvents (see 2026-04-21 20:45 Change Log for the
  // reconnect-loop bug this guards against).
  const terminalRef = useRef(false);

  useEffect(() => {
    setEvents([]);
    seqRef.current = 0;
    setConnected(false);
    setTerminal(false);
    terminalRef.current = false;
    if (!jobId) return;

    let cancelled = false;
    let ws: WebSocket | null = null;
    let retry = 0;
    // §Ship 21e: capture the reconnect timer so cleanup clears it.
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled || terminalRef.current) return;
      ws = new WebSocket(jobEventsWsUrl(jobId));

      ws.onopen = () => {
        retry = 0;
        setConnected(true);
      };

      ws.onmessage = (msg) => {
        let ev: JobEvent;
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
          if (ev.seq <= seqRef.current) return;
          seqRef.current = ev.seq;
        }
        setEvents((xs) => {
          // Cap at 500 events — these jobs emit 10-30 log_lines tops,
          // but a broken sculptor loop could flood; bound memory.
          const next = [...xs, ev];
          if (next.length > 500) next.splice(0, next.length - 500);
          return next;
        });
      };

      ws.onclose = () => {
        setConnected(false);
        if (cancelled || terminalRef.current) return;
        const delay = Math.min(8000, 250 * Math.pow(2, retry));
        retry++;
        reconnectTimer = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        // Handled by onclose.
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
  }, [jobId]);

  return { events, connected, terminal };
}

/** POST /jobs/{jobId}/stop — cooperative cancel (M7 Phase 6). Used by
 * the ActiveJobsIndicator's stop button. Invalidates the job + the
 * project's jobs list so the indicator disappears on success. */
export function useStopJob(projectSlug?: string) {
  const qc = useQueryClient();
  return useMutation<JobSummary, Error, string>({
    mutationFn: (jobId) => stopJob(jobId),
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: qk.job(job.job_id) });
      if (projectSlug) {
        qc.invalidateQueries({ queryKey: qk.projectJobs(projectSlug) });
      }
    },
  });
}
