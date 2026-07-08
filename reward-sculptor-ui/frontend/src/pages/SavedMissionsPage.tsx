import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Badge, Btn, EmptyState, Modal } from "@/components/rs/primitives";
import { useDeleteSaved, useSaved, useSavedEntry } from "@/hooks/useSaved";
import { ApiError, savedFileUrl } from "@/lib/api";
import { formatRelative } from "@/lib/utils";
import type { SavedEntryDetail, SavedEntrySummary } from "@/lib/types";

function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${Math.max(1, Math.round(n / 1e3))} KB`;
  return `${n} B`;
}

/** Saved-missions library — a DISK viewer over the durable archive.
 *  Deliberately NOT wired to the WS/job-coupled mission dialog: these
 *  entries are read-only snapshots that survive backend restarts. */
export default function SavedMissionsPage() {
  const { data, isLoading, error } = useSaved();
  const [openEntry, setOpenEntry] = useState<string | null>(null);
  const entries = data ?? [];

  return (
    <div className="rs-scroll">
      <div className="rs-pad rs-vgap-16" style={{ maxWidth: 1320 }}>
        <div>
          <div className="rs-eyebrow">durable archive · survives restarts</div>
          <h2 className="rs-h2" style={{ marginTop: 6 }}>Saved Missions</h2>
          <p className="rs-sub" style={{ margin: "6px 0 0", maxWidth: 640 }}>
            Missions snapshotted into a durable store — rollout videos, reward
            code, metrics and the built report, kept even if the source project
            is deleted. Open one to review it read-only.
          </p>
        </div>

        {isLoading && <p className="rs-sub">Loading saved missions…</p>}
        {error && (
          <div className="rs-banner err">
            <Icon name="alert-triangle" size={17} />
            <span className="rs-grow">Could not load saved missions: {(error as Error).message}</span>
          </div>
        )}
        {!isLoading && !error && entries.length === 0 && (
          <div className="rs-card">
            <EmptyState
              icon="package"
              title="No saved missions yet"
              sub="Save a mission from its detail view to snapshot it here. Missions also auto-archive as they run."
            />
          </div>
        )}

        {entries.length > 0 && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
              gap: 16,
            }}
          >
            {entries.map((e) => (
              <SavedCard key={e.entry_id} entry={e} onOpen={() => setOpenEntry(e.entry_id)} />
            ))}
          </div>
        )}
      </div>

      {openEntry && (
        <SavedDetailModal entryId={openEntry} onClose={() => setOpenEntry(null)} />
      )}
    </div>
  );
}

// ── Card ──────────────────────────────────────────────────────────────
function SavedCard({ entry, onOpen }: { entry: SavedEntrySummary; onOpen: () => void }) {
  const keptTotal = entry.stages.reduce((acc, s) => acc + (s.n_kept_checkpoints ?? 0), 0);
  return (
    <button
      onClick={onOpen}
      className="rs-card"
      style={{
        textAlign: "left", cursor: "pointer", padding: 0, width: "100%",
        border: "1px solid var(--hairline)", background: "var(--surface-card)",
        display: "flex", flexDirection: "column",
      }}
    >
      <SavedThumb entry={entry} />
      <div className="rs-card-pad rs-vgap-8" style={{ flex: 1 }}>
        <div style={{ fontSize: 13.5, fontWeight: 600, lineHeight: 1.4, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }} title={entry.goal}>
          {entry.goal}
        </div>
        <div className="rs-flex rs-gap-6 rs-wrap" style={{ fontSize: 11 }}>
          <code className="mono rs-sub">{entry.project_slug}/{entry.mission_slug}</code>
        </div>
        <div className="rs-flex rs-gap-6 rs-wrap">
          {entry.stages.map((s) => (
            <Badge key={s.name} status={s.status} label={s.name} />
          ))}
        </div>
        <div className="rs-flex rs-gap-12 rs-wrap rs-sub" style={{ fontSize: 11 }}>
          <span>{formatRelative(entry.created_at)}</span>
          <span className="rs-num">{fmtBytes(entry.total_bytes)}</span>
          <span className="rs-num" title="kept checkpoints across all stages">
            {keptTotal} ckpt{keptTotal === 1 ? "" : "s"}
          </span>
        </div>
      </div>
    </button>
  );
}

function SavedThumb({ entry }: { entry: SavedEntrySummary }) {
  const hasThumb = !!entry.thumbnail_video;
  return (
    <div
      style={{
        aspectRatio: "16/9", background: "#16150f", borderTopLeftRadius: "var(--radius-lg)",
        borderTopRightRadius: "var(--radius-lg)", overflow: "hidden",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}
    >
      {hasThumb ? (
        <video
          src={savedFileUrl(entry.entry_id, entry.thumbnail_video!)}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
          muted
          playsInline
          preload="metadata"
        >
          <track kind="captions" />
        </video>
      ) : (
        <Icon name="package" size={28} color="var(--rs-muted)" />
      )}
    </div>
  );
}

// ── Detail (read-only Modal) ──────────────────────────────────────────
function SavedDetailModal({ entryId, onClose }: { entryId: string; onClose: () => void }) {
  const detail = useSavedEntry(entryId);
  const del = useDeleteSaved();
  const d = detail.data;

  const onDelete = () => {
    const ok = window.confirm(
      `Move this saved mission to Trash? It stays recoverable from Settings → Trash.`,
    );
    if (!ok) return;
    del.mutate(entryId, {
      onSuccess: () => {
        toast.success("Moved to Trash", { description: "Recoverable from Settings → Trash." });
        onClose();
      },
      onError: (err) => {
        const msg = err instanceof ApiError ? err.problem.detail ?? err.problem.title : (err as Error).message;
        toast.error("Could not delete", { description: msg });
      },
    });
  };

  return (
    <Modal
      wide
      icon="package"
      title={d?.goal ?? "Saved mission"}
      subtitle={
        d ? (
          <span className="mono">
            {d.project_slug}/{d.mission_slug} · {fmtBytes(d.total_bytes)} · saved {formatRelative(d.created_at)}
          </span>
        ) : (
          "Loading…"
        )
      }
      onClose={onClose}
      footer={
        <>
          <Btn
            kind="danger"
            icon={del.isPending ? "loader" : "trash"}
            disabled={del.isPending || !d}
            onClick={onDelete}
          >
            Delete
          </Btn>
          <Btn kind="primary" onClick={onClose}>Close</Btn>
        </>
      }
    >
      {detail.isLoading && <p className="rs-sub" style={{ fontSize: 13 }}>Loading saved mission…</p>}
      {detail.error && (
        <div className="rs-banner err">
          <Icon name="alert-triangle" size={17} />
          <span className="rs-grow">{(detail.error as Error).message}</span>
        </div>
      )}
      {d && (
        <div className="rs-vgap-16">
          {d.stages.map((s) => (
            <SavedStageSection key={s.name} entryId={entryId} stage={s} />
          ))}
          <SavedReport entryId={entryId} />
        </div>
      )}
    </Modal>
  );
}

function SavedStageSection({
  entryId, stage,
}: {
  entryId: string;
  stage: SavedEntryDetail["stages"][number];
}) {
  // Newest video is the most representative rollup for the stage.
  const videos = stage.videos ?? [];
  const [picked, setPicked] = useState<number>(videos.length > 0 ? videos.length - 1 : 0);
  const activeVideo = videos[Math.min(picked, videos.length - 1)];

  return (
    <section
      style={{
        border: "1px solid var(--hairline)",
        borderRadius: "var(--radius-md)",
        background: "var(--surface-strong)",
        padding: 13,
      }}
    >
      <div className="rs-flex rs-wrap rs-gap-8" style={{ alignItems: "center", marginBottom: 10 }}>
        <Badge status={stage.status} />
        <span className="mono" style={{ fontSize: 12, fontWeight: 600 }}>{stage.name}</span>
        {stage.best_metric != null && (
          <span className="rs-sub rs-num" style={{ fontSize: 11.5 }}>best {stage.best_metric.toFixed(3)}</span>
        )}
        {typeof stage.n_iters === "number" && (
          <span className="rs-sub" style={{ fontSize: 11.5 }}>{stage.n_iters} iter{stage.n_iters === 1 ? "" : "s"}</span>
        )}
        {(stage.kept_checkpoints ?? []).length > 0 && (
          <span className="rs-sub" style={{ fontSize: 11.5 }}>
            kept {stage.kept_checkpoints.map((c) => `iter ${c.iter} (${c.reason})`).join(", ")}
          </span>
        )}
      </div>

      {activeVideo ? (
        <div style={{ border: "1px solid var(--hairline)", borderRadius: "var(--radius-md)", overflow: "hidden" }}>
          <div className="rs-log-bar" style={{ gap: 8 }}>
            <Icon name="video" size={13} color="var(--rs-muted)" />
            <span style={{ fontSize: 11.5, fontWeight: 600 }}>rollout</span>
            <span className="rs-grow" />
            {videos.length > 1 && (
              <div className="rs-select" style={{ display: "flex" }}>
                <select
                  value={picked}
                  onChange={(e) => setPicked(Number(e.target.value))}
                  aria-label={`${stage.name} rollout`}
                >
                  {videos.map((_, i) => (
                    <option key={i} value={i}>rollout {i + 1} of {videos.length}</option>
                  ))}
                </select>
              </div>
            )}
            <a href={savedFileUrl(entryId, activeVideo)} download className="rs-btn rs-btn-quiet rs-btn-xs">
              <Icon name="download" size={13} />MP4
            </a>
          </div>
          <video
            key={activeVideo}
            src={savedFileUrl(entryId, activeVideo)}
            style={{ width: "100%", aspectRatio: "16/9", background: "#16150f", display: "block" }}
            controls
            playsInline
            preload="metadata"
          >
            <track kind="captions" />
          </video>
        </div>
      ) : (
        <p className="rs-sub" style={{ fontSize: 11.5, margin: 0 }}>No rollout video archived for this stage.</p>
      )}
    </section>
  );
}

// ── Archived report markdown (only rendered if present) ───────────────
function SavedReport({ entryId }: { entryId: string }) {
  // The manifest doesn't carry a report path; probe the conventional
  // location and render only on a hit (404 = no report archived).
  const report = useQuery<string>({
    queryKey: [...qkSavedReport(entryId)],
    queryFn: async () => {
      const r = await fetch(savedFileUrl(entryId, "reports/final_report.md"));
      if (!r.ok) return "";
      return await r.text();
    },
    staleTime: 60_000,
    retry: false,
  });
  const md = (report.data ?? "").trim();
  if (!md) return null;
  return (
    <section
      className="rs-report"
      style={{
        border: "1px solid var(--hairline)",
        borderRadius: "var(--radius-md)",
        background: "var(--surface-card)",
        padding: "20px 24px",
      }}
    >
      <div className="rs-card-title" style={{ marginBottom: 12 }}>
        <Icon name="file-text" size={16} />Archived report
      </div>
      <div className="rs-md">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{md}</ReactMarkdown>
      </div>
    </section>
  );
}

function qkSavedReport(entryId: string) {
  return ["saved", entryId, "report", "md"] as const;
}
