import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Icon } from "@/components/rs/icon";
import { Btn, EmptyState, Modal } from "@/components/rs/primitives";
import {
  useAttachStageReference,
  useReferenceIndex,
  useReferenceSearch,
} from "@/hooks/useReferences";
import { ApiError, getReferencePreviewUrl } from "@/lib/api";
import type { RefIndexRow, RefMatch } from "@/lib/types";

// Search-as-you-type debounce. Deterministic endpoint (llm=0) is cheap,
// but we still don't want a fetch per keystroke.
const SEARCH_DEBOUNCE_MS = 250;

function fmtDuration(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "—";
  const m = Math.floor(s / 60);
  const rem = Math.round(s - m * 60);
  return `${m}:${String(rem).padStart(2, "0")}`;
}

/** Normalize a search hit or a plain index row into the shape the row
 *  renderer + preview pane need. Search hits carry score/confidence;
 *  browse rows (empty query) don't. */
type PickerRow = {
  clip_id: string;
  text: string;
  tier: string;
  license: string;
  duration_s: number;
  score?: number;
  match_confidence?: number | null;
};

function toRow(m: RefMatch): PickerRow {
  return {
    clip_id: m.clip_id, text: m.text, tier: m.tier, license: m.license,
    duration_s: m.duration_s, score: m.score, match_confidence: m.match_confidence,
  };
}
function indexToRow(r: RefIndexRow): PickerRow {
  return { clip_id: r.clip_id, text: r.text, tier: r.tier, license: r.license, duration_s: r.duration_s };
}

/** Small keyframe-strip preview. Hides itself on 404 (no preview.png
 *  for this clip — decision 8 says preview generation must never block
 *  ingest, so absence is an expected, non-error state). */
function PreviewImage({ clipId }: { clipId: string }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [clipId]);
  if (failed) return null;
  return (
    <img
      src={getReferencePreviewUrl(clipId)}
      alt={`${clipId} keyframe preview`}
      onError={() => setFailed(true)}
      style={{
        width: "100%", maxHeight: 90, objectFit: "contain",
        borderRadius: "var(--radius-sm)", border: "1px solid var(--hairline)",
        background: "var(--canvas-soft)",
      }}
    />
  );
}

function ResultRow({
  row, selected, onSelect,
}: { row: PickerRow; selected: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      style={{
        display: "flex", alignItems: "center", gap: 10, width: "100%",
        textAlign: "left", padding: "8px 10px", cursor: "pointer",
        border: "1px solid " + (selected ? "var(--rs-primary)" : "var(--hairline)"),
        background: selected ? "rgba(245,78,0,0.05)" : "var(--surface-strong)",
        borderRadius: "var(--radius-md)", font: "inherit", color: "inherit",
      }}
    >
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ fontSize: 12.5, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {row.text}
        </div>
        <div className="rs-sub" style={{ marginTop: 3, fontSize: 10.5, display: "flex", flexWrap: "wrap", gap: 8 }}>
          <span>{fmtDuration(row.duration_s)}</span>
          <span className="rs-badge slate" style={{ fontSize: 9 }}>tier {row.tier}</span>
          <span>{row.license}</span>
          {row.score != null && <span>score {row.score.toFixed(2)}</span>}
          {row.match_confidence != null && <span>confidence {row.match_confidence.toFixed(2)}</span>}
        </div>
      </div>
      {selected && <Icon name="check" size={14} color="var(--rs-primary)" />}
    </button>
  );
}

export function ReferencePickerDialog({
  slug,
  missionSlug,
  stageName,
  currentClipId,
  onClose,
}: {
  slug: string;
  missionSlug: string;
  stageName: string;
  /** Currently attached clip, if any — pre-selects it so re-opening the
   *  picker on an already-attached stage shows the existing choice. */
  currentClipId?: string | null;
  onClose: () => void;
}) {
  const [queryInput, setQueryInput] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  useEffect(() => {
    const id = setTimeout(() => setDebouncedQuery(queryInput), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [queryInput]);

  const [selectedClipId, setSelectedClipId] = useState<string | null>(currentClipId ?? null);

  const trimmed = debouncedQuery.trim();
  const search = useReferenceSearch(trimmed, { enabled: trimmed.length > 0 });
  const browse = useReferenceIndex({ enabled: trimmed.length === 0 });

  const rows: PickerRow[] = trimmed.length > 0
    ? (search.data ?? []).map(toRow)
    : (browse.data ?? []).map(indexToRow);
  const isLoading = trimmed.length > 0 ? search.isLoading : browse.isLoading;
  const isError = trimmed.length > 0 ? search.isError : browse.isError;
  const libraryEmpty = trimmed.length === 0 && !browse.isLoading && !browse.isError && rows.length === 0;

  const attach = useAttachStageReference(slug);

  const doAttach = () => {
    if (!selectedClipId) return;
    attach.mutate(
      { missionSlug, stageName, clipId: selectedClipId },
      {
        onSuccess: () => {
          toast.success("Reference attached", { description: selectedClipId });
          onClose();
        },
        onError: (err) => {
          if (err instanceof ApiError && err.status === 409) {
            toast.error("Mission is busy", {
              description: "Wait for the running job to finish before attaching a reference.",
            });
            return;
          }
          const detail = err instanceof ApiError ? (err.problem.detail ?? err.problem.title) : err.message;
          toast.error("Could not attach reference", { description: String(detail) });
        },
      },
    );
  };

  return (
    <Modal
      icon="video"
      title="Pick a reference clip"
      subtitle={`Stage ${stageName}`}
      onClose={onClose}
      footer={
        <>
          <Btn kind="quiet" onClick={onClose} disabled={attach.isPending}>Cancel</Btn>
          <Btn
            kind="primary"
            icon={attach.isPending ? "loader" : "check"}
            onClick={doAttach}
            disabled={!selectedClipId || attach.isPending}
          >
            {attach.isPending ? "Attaching…" : "Attach"}
          </Btn>
        </>
      }
    >
      <div className="rs-flex rs-gap-8" style={{ background: "var(--canvas-soft)", border: "1px solid var(--hairline-strong)", borderRadius: "var(--radius-md)", padding: "0 11px", height: 32, marginBottom: 12 }}>
        <Icon name="search" size={14} color="var(--rs-muted)" />
        <input
          value={queryInput}
          onChange={(e) => setQueryInput(e.target.value)}
          placeholder='Search, e.g. "get up off the ground"'
          aria-label="Search reference clips"
          autoFocus
          style={{ border: 0, background: "none", outline: "none", fontSize: 13, width: "100%", color: "var(--ink)" }}
        />
      </div>

      {isLoading && <p className="rs-sub">Searching…</p>}
      {isError && (
        <div className="rs-banner err">
          <Icon name="alert-triangle" size={17} />
          <span className="rs-grow">Could not load reference clips.</span>
        </div>
      )}
      {libraryEmpty && (
        <EmptyState
          icon="video"
          title="Reference library is empty"
          sub="No clips have been ingested yet. Run `sculpt refs ingest` to populate it."
        />
      )}
      {!isLoading && !isError && !libraryEmpty && trimmed.length > 0 && rows.length === 0 && (
        <EmptyState icon="search" title="No matches" sub={`Nothing matches "${trimmed}".`} />
      )}

      {rows.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 260, overflowY: "auto" }}>
          {rows.map((row) => (
            <ResultRow
              key={row.clip_id}
              row={row}
              selected={selectedClipId === row.clip_id}
              onSelect={() => setSelectedClipId(row.clip_id)}
            />
          ))}
        </div>
      )}

      {selectedClipId && (
        <div style={{ marginTop: 12, borderTop: "1px solid var(--hairline)", paddingTop: 12 }}>
          <div className="rs-sub" style={{ fontSize: 10.5, marginBottom: 6 }}>Preview</div>
          <PreviewImage clipId={selectedClipId} />
        </div>
      )}
    </Modal>
  );
}
