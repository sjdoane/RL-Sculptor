import { useEffect, useState } from "react";
import { toast } from "sonner";

import { ComposeMotionDialog } from "@/components/ComposeMotionDialog";
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

/** Page size for the no-query browse listing. */
const BROWSE_PAGE = 40;

export function ReferencePickerDialog({
  slug,
  missionSlug,
  stageName,
  currentClipId,
  initialQuery,
  robot = "g1",
  onPick,
  onClose,
}: {
  slug: string;
  missionSlug?: string;
  stageName?: string;
  /** Currently attached clip, if any — pre-selects it so re-opening the
   *  picker on an already-attached stage shows the existing choice. */
  currentClipId?: string | null;
  /** Stage goal text used to seed the search on open. With a 6k-clip
   *  library, the no-query browse listing leads with alphabetical
   *  noise (0000_motorcycle…) — opening pre-searched on the stage's
   *  own goal surfaces relevant clips immediately. */
  initialQuery?: string;
  /** Reference-library embodiment namespace. The normal run picker derives
   *  this from project metadata; mission callers keep the historical g1
   *  default unless they pass a different robot. */
  robot?: string;
  /** Standalone selection mode. When provided, selecting a clip returns it
   *  to the caller instead of mutating a mission stage. */
  onPick?: (selection: { clipId: string; robot: string }) => void;
  onClose: () => void;
}) {
  const [queryInput, setQueryInput] = useState(initialQuery ?? "");
  const [debouncedQuery, setDebouncedQuery] = useState(initialQuery ?? "");
  useEffect(() => {
    const id = setTimeout(() => setDebouncedQuery(queryInput), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [queryInput]);

  const [selectedClipId, setSelectedClipId] = useState<string | null>(currentClipId ?? null);
  const [composing, setComposing] = useState(false);
  const [browseOffset, setBrowseOffset] = useState(0);

  const trimmed = debouncedQuery.trim();
  const search = useReferenceSearch(trimmed, {
    robot, enabled: trimmed.length > 0,
  });
  const browse = useReferenceIndex({
    robot, enabled: trimmed.length === 0,
    limit: BROWSE_PAGE, offset: browseOffset,
  });
  // A new query starts from the top of its own result set.
  useEffect(() => { setBrowseOffset(0); }, [trimmed]);

  const rows: PickerRow[] = trimmed.length > 0
    ? (search.data ?? []).map(toRow)
    : (browse.data?.rows ?? []).map(indexToRow);
  const isLoading = trimmed.length > 0 ? search.isLoading : browse.isLoading;
  const isError = trimmed.length > 0 ? search.isError : browse.isError;
  const libraryEmpty = trimmed.length === 0 && !browse.isLoading && !browse.isError && rows.length === 0;

  const attach = useAttachStageReference(slug);
  const isStandalone = onPick !== undefined;
  const isCommitting = !isStandalone && attach.isPending;

  const doAttach = () => {
    if (!selectedClipId) return;
    if (onPick) {
      onPick({ clipId: selectedClipId, robot });
      onClose();
      return;
    }
    if (!missionSlug || !stageName) {
      toast.error("Reference destination is missing");
      return;
    }
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
      subtitle={stageName ? `Stage ${stageName}` : `Motion prior · ${robot}`}
      onClose={onClose}
      footer={
        <>
          {/* No single clip covers every goal. Composing is the path when the
              motion exists only as phases spread across several clips, so the
              affordance belongs here — at the moment the search comes up short. */}
          <Btn
            kind="quiet"
            icon="plus"
            onClick={() => setComposing(true)}
            disabled={isCommitting}
          >
            Compose novel
          </Btn>
          <span className="rs-grow" />
          <Btn kind="quiet" onClick={onClose} disabled={isCommitting}>Cancel</Btn>
          <Btn
            kind="primary"
            icon={isCommitting ? "loader" : "check"}
            onClick={doAttach}
            disabled={!selectedClipId || isCommitting}
          >
            {isCommitting
              ? "Attaching…"
              : isStandalone ? "Use motion" : "Attach"}
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
        <EmptyState
          icon="search"
          title="No matches"
          sub={`Nothing matches "${trimmed}". If this motion exists only as separate phases, compose it from them.`}
        />
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

      {/* Say how much of the library is on screen. The browse listing used to
          be a silent `rows[:10]` of ~6015, which read as "this is the
          library" — including right after composing a clip that was not in
          those ten. */}
      {trimmed.length === 0 && browse.data && browse.data.total > rows.length && (
        <div
          className="rs-flex rs-gap-8"
          style={{ marginTop: 8, alignItems: "center", fontSize: 11 }}
        >
          <span className="rs-sub">
            {browseOffset + 1}–{browseOffset + rows.length} of{" "}
            {browse.data.total} clips
            {browse.data.facets.composed > 0 && (
              <> · {browse.data.facets.composed} composed</>
            )}
          </span>
          <span className="rs-grow" />
          <Btn
            kind="quiet" size="xs" icon="chevron-left"
            disabled={browseOffset === 0}
            onClick={() => setBrowseOffset((o) => Math.max(0, o - BROWSE_PAGE))}
          >
            Prev
          </Btn>
          <Btn
            kind="quiet" size="xs" icon="chevron-right"
            disabled={browseOffset + rows.length >= browse.data.total}
            onClick={() => setBrowseOffset((o) => o + BROWSE_PAGE)}
          >
            Next
          </Btn>
        </div>
      )}

      {selectedClipId && (
        <div style={{ marginTop: 12, borderTop: "1px solid var(--hairline)", paddingTop: 12 }}>
          <div className="rs-sub" style={{ fontSize: 10.5, marginBottom: 6 }}>Preview</div>
          <PreviewImage clipId={selectedClipId} />
        </div>
      )}

      {composing && (
        <ComposeMotionDialog
          robot={robot}
          onClose={() => setComposing(false)}
          onComposed={(clipId) => {
            // The composite is almost always the clip the user came here to
            // attach, so select it and put it in the search box rather than
            // making them hunt for it in a 6k-row listing.
            setSelectedClipId(clipId);
            setQueryInput(clipId);
          }}
        />
      )}
    </Modal>
  );
}
