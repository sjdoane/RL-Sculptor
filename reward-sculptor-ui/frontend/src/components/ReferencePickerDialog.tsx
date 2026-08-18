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
import { ApiError, getReferenceClipUrl, getReferencePreviewUrl } from "@/lib/api";
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
  /** One-line LLM rationale; only the reranked path produces one. */
  reason?: string | null;
};

function toRow(m: RefMatch): PickerRow {
  return {
    clip_id: m.clip_id, text: m.text, tier: m.tier, license: m.license,
    duration_s: m.duration_s, score: m.score, match_confidence: m.match_confidence,
    reason: m.reason,
  };
}
function indexToRow(r: RefIndexRow): PickerRow {
  return { clip_id: r.clip_id, text: r.text, tier: r.tier, license: r.license, duration_s: r.duration_s };
}

/** Small keyframe-strip preview. Hides itself on 404 (no preview.png
 *  for this clip — decision 8 says preview generation must never block
 *  ingest, so absence is an expected, non-error state). */
function PreviewImage({ robot, clipId }: { robot: string; clipId: string }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [robot, clipId]);
  if (failed) return null;
  return (
    <img
      src={getReferencePreviewUrl(robot, clipId)}
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
        {/* The rerank's own reason for the ranking. It was being returned
            and dropped, which made a reranked list look like an unexplained
            reshuffle of the deterministic one. */}
        {row.reason && (
          <div
            className="rs-sub"
            style={{ marginTop: 3, fontSize: 10.5, fontStyle: "italic", whiteSpace: "normal" }}
          >
            {row.reason}
          </div>
        )}
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
  robot,
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
  /** Canonical reference-library embodiment namespace resolved from project
   *  metadata. Every caller must pass it explicitly; catalog slugs and task
   *  names are not reference namespaces. */
  robot: string;
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

  // Browse filters. Every one is a query param the endpoint already
  // accepted and nothing sent — which is why a 6k-clip library could only
  // be paged through in one fixed order.
  const [tier, setTier] = useState("");
  const [label, setLabel] = useState("");
  const [sort, setSort] = useState<"recent" | "duration" | "name">("recent");
  const [composedOnly, setComposedOnly] = useState(false);
  const [maxDur, setMaxDur] = useState("");

  const trimmed = debouncedQuery.trim();
  const maxDurationS = Number.parseFloat(maxDur);
  // The exact query the user asked to rerank. Comparing against the live
  // query (rather than holding a boolean) means editing the text drops
  // straight back to the cheap deterministic path — an LLM call per
  // keystroke is what the endpoint's llm=0 default exists to prevent.
  const [rerankedQuery, setRerankedQuery] = useState<string | null>(null);
  const reranked = rerankedQuery !== null && rerankedQuery === trimmed;
  const search = useReferenceSearch(trimmed, {
    robot, enabled: trimmed.length > 0, useLlm: reranked,
  });
  const browse = useReferenceIndex({
    robot, enabled: trimmed.length === 0,
    limit: BROWSE_PAGE, offset: browseOffset,
    tier: tier || undefined,
    label: label || undefined,
    composed: composedOnly ? true : undefined,
    maxDurationS: Number.isFinite(maxDurationS) ? maxDurationS : undefined,
    sort,
  });
  const filtered = !!(tier || label || composedOnly || maxDur.trim());
  // A new query — or a new filter — starts from the top of its own result
  // set. Keeping the old offset would page past the end of a smaller one.
  useEffect(() => {
    setBrowseOffset(0);
  }, [trimmed, tier, label, sort, composedOnly, maxDur]);

  // Facets come back with every browse response and describe the WHOLE
  // library for this robot, not the filtered page — so the options don't
  // vanish as you narrow. Keep the last non-empty set so they don't flicker
  // to empty while a filtered page is in flight.
  const facets = browse.data?.facets;
  const tierOptions = Object.entries(facets?.tiers ?? {}).sort();
  // Labels are derived from clip ids, so the raw top-of-list is filename
  // debris — `120`, `100`, `1`, `f`. Those partition the library without
  // describing anything, so they aren't offered; `novel`, `composed`,
  // `locomotion` and friends survive.
  const labelOptions = Object.entries(facets?.labels ?? {})
    .filter(([l]) => l.length > 2 && !/^\d+$/.test(l))
    .sort((a, b) => b[1] - a[1]).slice(0, 24);
  // One distinct tier means the control can only ever be a no-op. Today the
  // whole corpus is tier K (kinematic-only); it earns its place as soon as a
  // clip is dynamics-certified, so the control appears then rather than
  // sitting there inert.
  const showTier = tierOptions.length > 1 || !!tier;

  const rows: PickerRow[] = trimmed.length > 0
    ? (search.data ?? []).map(toRow)
    : (browse.data?.rows ?? []).map(indexToRow);
  const isLoading = trimmed.length > 0 ? search.isLoading : browse.isLoading;
  const isError = trimmed.length > 0 ? search.isError : browse.isError;
  const browseEmpty = trimmed.length === 0 && !browse.isLoading && !browse.isError && rows.length === 0;
  // An empty filtered page is not an empty library. Saying "run `sculpt refs
  // ingest`" to someone who just narrowed to Tier A would send them to fix
  // something that isn't broken.
  const libraryEmpty = browseEmpty && !filtered;
  const filteredEmpty = browseEmpty && filtered;

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
      subtitle={stageName
        ? `Stage ${stageName} · ${robot}`
        : `Motion prior · ${robot}`}
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
        {/* Semantic search runs deterministic so it can keep up with typing.
            The LLM rerank was therefore unreachable — this asks for it once,
            for the query on screen, and each hit comes back with a stated
            reason. */}
        {trimmed.length > 0 && (
          <Btn
            kind={reranked ? "primary" : "quiet"}
            size="xs"
            icon={reranked && search.isFetching ? "loader" : "sparkles"}
            disabled={reranked && search.isFetching}
            title="Re-rank these results with an LLM, with a one-line reason per clip."
            onClick={() => setRerankedQuery(reranked ? null : trimmed)}
          >
            {reranked
              ? (search.isFetching ? "Ranking…" : "AI-ranked")
              : "AI rank"}
          </Btn>
        )}
      </div>

      {/* Filters apply to browsing, not to the semantic search endpoint —
          that one ranks by embedding similarity and takes no facets. Rather
          than let a set filter silently stop applying the moment you type,
          the bar disables itself and says so. */}
      <div
        className="rs-flex rs-gap-6"
        style={{ flexWrap: "wrap", alignItems: "center", marginBottom: 10 }}
      >
        <Icon name="filter" size={13} color="var(--rs-muted)" />
        <div className="rs-select">
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as typeof sort)}
            disabled={trimmed.length > 0}
            aria-label="Sort clips"
          >
            <option value="recent">Newest first</option>
            <option value="duration">Longest first</option>
            <option value="name">By name</option>
          </select>
        </div>
        {showTier && (
          <div className="rs-select">
            <select
              value={tier}
              onChange={(e) => setTier(e.target.value)}
              disabled={trimmed.length > 0}
              aria-label="Filter by retarget tier"
            >
              <option value="">Any tier</option>
              {tierOptions.map(([t, n]) => (
                <option key={t} value={t}>Tier {t} ({n})</option>
              ))}
            </select>
          </div>
        )}
        <div className="rs-select">
          <select
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            disabled={trimmed.length > 0}
            aria-label="Filter by label"
          >
            <option value="">Any label</option>
            {labelOptions.map(([l, n]) => (
              <option key={l} value={l}>{l} ({n})</option>
            ))}
          </select>
        </div>
        <input
          value={maxDur}
          onChange={(e) => setMaxDur(e.target.value)}
          disabled={trimmed.length > 0}
          inputMode="decimal"
          placeholder="max s"
          aria-label="Maximum clip duration in seconds"
          style={{
            width: 66, height: 28, fontSize: 12, padding: "0 8px",
            background: "var(--surface-card)", color: "var(--ink)",
            border: "1px solid var(--hairline)",
            borderRadius: "var(--radius-sm)",
          }}
        />
        <Btn
          kind={composedOnly ? "primary" : "quiet"}
          size="xs"
          icon="sparkles"
          disabled={trimmed.length > 0}
          onClick={() => setComposedOnly((v) => !v)}
        >
          Composed{facets?.composed ? ` (${facets.composed})` : ""}
        </Btn>
        {filtered && trimmed.length === 0 && (
          <Btn
            kind="quiet" size="xs" icon="x"
            onClick={() => {
              setTier(""); setLabel(""); setComposedOnly(false); setMaxDur("");
            }}
          >
            Clear
          </Btn>
        )}
        {trimmed.length > 0 && (
          <span className="rs-sub" style={{ fontSize: 11 }}>
            {filtered
              ? "Filters are paused — semantic search ranks the whole library."
              : "Clear the search box to filter and sort the library."}
          </span>
        )}
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
      {filteredEmpty && (
        <EmptyState
          icon="filter"
          title="No clips match these filters"
          sub={`The library has ${facets?.total ?? 0} ${robot} clips. Widen or clear the filters to see them.`}
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
      {trimmed.length === 0 && browse.data && rows.length > 0
        && (browse.data.total > rows.length || filtered) && (
        <div
          className="rs-flex rs-gap-8"
          style={{ marginTop: 8, alignItems: "center", fontSize: 11 }}
        >
          <span className="rs-sub">
            {browseOffset + 1}–{browseOffset + rows.length} of{" "}
            {browse.data.total} {filtered ? "matching" : ""} clips
            {/* When filtered, `total` is the match count — say what it was
                narrowed from, so the filter's effect is legible. */}
            {filtered && browse.data.facets.total > browse.data.total && (
              <> (of {browse.data.facets.total})</>
            )}
            {!filtered && browse.data.facets.composed > 0 && (
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
          <div
            className="rs-flex rs-gap-8"
            style={{ alignItems: "baseline", marginBottom: 6 }}
          >
            <span className="rs-sub" style={{ fontSize: 10.5 }}>Preview</span>
            <span className="rs-grow" />
            {/* A composed clip existed only inside the app: there was no way
                to get its arrays out to inspect or archive them. */}
            <a
              className="rs-sub"
              href={getReferenceClipUrl(robot, selectedClipId)}
              download={`${selectedClipId}.npz`}
              style={{ fontSize: 10.5, display: "inline-flex", alignItems: "center", gap: 4 }}
            >
              <Icon name="download" size={12} />
              clip.npz
            </a>
          </div>
          <PreviewImage robot={robot} clipId={selectedClipId} />
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
