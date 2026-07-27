"""Deterministic root_z hysteresis segmentation for multi-repetition
clips (§R1_BUILD_SPEC decision 6, refined 2026-07-09 per the "get-up
segments must start lying down" fix).

Long recordings like `fallAndGetUp1_subject1` (thousands of frames,
several fall/recover repetitions) are useless as a single RSI reference
— the airborne/recovery statistics of one repetition get diluted by five
others. This module splits such a clip into per-repetition segments
using a two-threshold (hysteresis) state machine on `root_pos_z`, so a
single brief dip below the "down" threshold doesn't spuriously end a
segment, and noise near either threshold doesn't create spurious
segments.

State machine (§decision 6, exact thresholds):
    "standing" when z > 0.60 m sustained >= 1.0 s
    "down"     when z < 0.35 m sustained >= 0.5 s
A segment runs from a down-interval to the NEXT sustained-standing
interval, with a 2 s minimum length (segments shorter than that are
dropped — they're noise, not a real fall+recover cycle).

2026-07-09 fix — settled-start selection: measured against the real
library, the original rule (`down-interval start - 0.5s standing pad`)
produced get-up segments that START STANDING (the pre-fall pad) and
contain the fall itself, e.g. `fallandgetup1_subject1--seg00` started at
z=0.637 — fully upright. A torso-righting reference needs a segment that
starts at the SETTLED lying point, not mid-fall or pre-fall. The start
is now the first frame inside the down-interval where z is below
`DOWN_Z` AND the trailing `SETTLE_WINDOW_S` window is nearly static
(see `_find_settled_start`); there is no leading pad. The trailing
+/-`PAD_S` pad after sustained-standing is unchanged (still captures the
stood-up hold).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

STANDING_Z = 0.60
DOWN_Z = 0.35
STANDING_SUSTAIN_S = 1.0
DOWN_SUSTAIN_S = 0.5
PAD_S = 0.5
MIN_SEGMENT_S = 2.0

#: Settled-start detection (2026-07-09 fix). A frame counts as "settled
#: lying" when z is below DOWN_Z and the subject has stopped moving
#: vertically for a short trailing window — this is what distinguishes
#: "mid-fall, still dropping" from "on the ground, at rest". The window
#: length is deliberately short (0.3 s) so it catches the earliest
#: settled moment rather than requiring a long static hold (that's what
#: the down-interval's own DOWN_SUSTAIN_S already guarantees downstream).
SETTLE_WINDOW_S = 0.3
#: |dz/dt| threshold for "not moving", in m/s. Chosen well below a
#: normal stand<->crouch transition speed (roughly 0.5-1.5 m/s for a
#: human-scale get-up) but well above per-frame retargeting/mocap noise
#: (sub-mm at 30fps is << 0.01 m/s); 0.25 m/s at a 30fps-equivalent
#: sample rate is ~0.008 m/frame, comfortably above noise and
#: comfortably below active-motion speeds.
SETTLE_VEL_THRESHOLD_MPS = 0.25

#: QC gate thresholds (build spec item 2): a get-up segment must start
#: lying down and end standing, checked over the first/last 10% of the
#: segment (not single frames) so a single noisy sample can't flip the
#: verdict. Mirrors the ingest-time fall/getup motion-class thresholds
#: (DOWN_Z / STANDING_Z) but on windowed means, which is intentionally
#: slightly looser than the raw 0.35/0.60 hysteresis thresholds — a
#: 10%-window mean naturally pulls in a few already-rising/falling
#: frames near the boundary.
QC_START_WINDOW_FRACTION = 0.10
QC_START_MAX_MEAN_Z = 0.35
QC_END_WINDOW_FRACTION = 0.10
QC_END_MIN_MEAN_Z = 0.55


@dataclass(frozen=True)
class Segment:
    """A segmentation result in frame indices, half-open `[start, end)`."""

    start: int
    end: int

    @property
    def n_frames(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RejectedSegment:
    """A candidate segment that failed the start/end QC gate (build spec
    item 2). Carries the frame range it would have had, so a caller can
    still log/inspect it via the same rejects mechanism ingest uses."""

    start: int
    end: int
    reason: str


@dataclass(frozen=True)
class SegmentationResult:
    """`segment_by_root_z`'s full output: accepted segments plus any
    candidates that were rejected by the start/end QC gate (item 2).
    Rejections are informational, not fatal — callers decide what to do
    with them (e.g. `ingest`/`resegment` log each via
    `library.append_reject`)."""

    segments: list[Segment] = field(default_factory=list)
    rejected: list[RejectedSegment] = field(default_factory=list)


def _sustained_runs(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    """Half-open `[start, end)` index ranges where `mask` is True for at
    least `min_frames` consecutive frames."""
    runs: list[tuple[int, int]] = []
    n = mask.shape[0]
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) >= min_frames:
            runs.append((i, j))
        i = j
    return runs


def _find_settled_start(z: np.ndarray, d_start: int, d_end: int, fps: float) -> int:
    """Within down-interval `[d_start, d_end)`, find the first frame that
    is both below `DOWN_Z` and vertically at rest (settled lying), per
    the 2026-07-09 fix. Falls back to the interval's z-argmin frame if no
    frame satisfies both conditions (e.g. the subject bounces straight
    back up without ever settling) — the lowest point is still a much
    better "start lying down" proxy than the old pre-fall standing pad.
    """
    settle_win = max(1, int(round(SETTLE_WINDOW_S * fps)))
    # Per-step velocity threshold: SETTLE_VEL_THRESHOLD_MPS is a rate;
    # convert to a per-frame delta at this clip's actual fps.
    max_step_delta = SETTLE_VEL_THRESHOLD_MPS / fps

    for i in range(d_start, d_end):
        if z[i] >= DOWN_Z:
            continue
        win_start = i - settle_win
        if win_start < 0:
            # Not enough trailing history in the whole clip yet.
            continue
        window = z[win_start:i + 1]
        if window.shape[0] < settle_win + 1:
            # Full SETTLE_WINDOW_S of trailing history isn't available
            # yet — a short/partial window (e.g. right at the
            # standing->down transition, where the last standing frame
            # and the first down frame can look "still" together by
            # coincidence) must not count as settled.
            continue
        if np.all(np.abs(np.diff(window)) < max_step_delta):
            return i

    # Fallback: no settled frame in this down-interval — use the
    # z-argmin frame (closest thing to "on the ground" available).
    local = z[d_start:d_end]
    return d_start + int(np.argmin(local))


def _qc_segment(z: np.ndarray, start: int, end: int) -> Optional[str]:
    """Build-spec item 2 hard QC gate: a get-up segment must start lying
    down (start-window mean z < QC_START_MAX_MEAN_Z) and end standing
    (end-window mean z > QC_END_MIN_MEAN_Z), checked over the first/last
    QC_*_WINDOW_FRACTION of the segment. Returns a rejection reason
    string, or None if the segment passes."""
    n = end - start
    if n <= 0:
        return "empty segment"
    seg = z[start:end]
    start_win = max(1, int(round(n * QC_START_WINDOW_FRACTION)))
    end_win = max(1, int(round(n * QC_END_WINDOW_FRACTION)))
    start_mean = float(seg[:start_win].mean())
    end_mean = float(seg[-end_win:].mean())
    if not (start_mean < QC_START_MAX_MEAN_Z):
        return (
            f"start-window mean z={start_mean:.3f} >= "
            f"{QC_START_MAX_MEAN_Z} (segment does not start lying down)")
    if not (end_mean > QC_END_MIN_MEAN_Z):
        return (
            f"end-window mean z={end_mean:.3f} <= "
            f"{QC_END_MIN_MEAN_Z} (segment does not end standing)")
    return None


def segment_by_root_z(z: np.ndarray, fps: float) -> list[Segment]:
    """Segment a root-height trace into down->recover cycles.

    Deterministic, offline, no LLM. Returns only QC-accepted segments in
    frame order; overlapping/adjacent padded windows are NOT merged
    (§decision 6 says nothing about merging, and keeping cycles distinct
    is what a multi-rep RSI consumer wants — each is a complete
    fall+recover). Callers that also need rejection reasons (e.g.
    ingest/resegment, for logging) should call `segment_by_root_z_full`
    instead — this function is kept as the simple/back-compat entry
    point used by earlier callers and tests.
    """
    return segment_by_root_z_full(z, fps).segments


def segment_by_root_z_full(z: np.ndarray, fps: float) -> SegmentationResult:
    """Full segmentation: same algorithm as `segment_by_root_z`, but also
    returns QC-rejected candidates with reasons (build spec item 2)."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or z.size == 0:
        return SegmentationResult()
    fps = float(fps)
    standing_min = max(1, int(round(STANDING_SUSTAIN_S * fps)))
    down_min = max(1, int(round(DOWN_SUSTAIN_S * fps)))
    pad = int(round(PAD_S * fps))
    min_len = max(1, int(round(MIN_SEGMENT_S * fps)))

    standing_runs = _sustained_runs(z > STANDING_Z, standing_min)
    down_runs = _sustained_runs(z < DOWN_Z, down_min)
    if not down_runs or not standing_runs:
        return SegmentationResult()

    segments: list[Segment] = []
    rejected: list[RejectedSegment] = []
    for d_start, d_end in down_runs:
        # Next sustained-standing run that starts at/after this down run
        # ends — "down-interval -> next sustained-standing".
        nxt = next(
            (s for s in standing_runs if s[0] >= d_end), None)
        if nxt is None:
            continue
        # 2026-07-09 fix: start at the settled-lying frame inside the
        # down-interval, NOT `d_start - pad` (that was the pre-fall
        # standing pad — it made segments start upright and include the
        # fall). No leading pad at all. The trailing pad after
        # sustained-standing is unchanged.
        seg_start = _find_settled_start(z, d_start, d_end, fps)
        seg_end = min(z.size, nxt[1] + pad)
        if (seg_end - seg_start) < min_len:
            continue
        reason = _qc_segment(z, seg_start, seg_end)
        if reason is not None:
            rejected.append(RejectedSegment(seg_start, seg_end, reason))
            continue
        segments.append(Segment(seg_start, seg_end))
    return SegmentationResult(segments=segments, rejected=rejected)


def segment_clip(clip: dict) -> list[dict]:
    """Apply `segment_by_root_z` to a validated reference clip and slice
    every array-valued key (+meta preserved) into per-segment clip dicts.
    Callers are responsible for re-validating and persisting each slice
    with derived provenance (`parent_clip_id` / `frame_range`). QC-
    rejected candidates are silently dropped here — use `segment_clip_full`
    if rejection reasons need to be logged."""
    return segment_clip_full(clip).segments


@dataclass(frozen=True)
class ClipSegmentationResult:
    """`segment_clip_full`'s output: accepted per-segment clip dicts plus
    QC-rejected candidates (frame range + reason), for callers that log
    rejects (ingest, resegment)."""

    segments: list[dict] = field(default_factory=list)
    rejected: list[RejectedSegment] = field(default_factory=list)


def segment_clip_full(clip: dict) -> ClipSegmentationResult:
    """Like `segment_clip`, but also returns QC-rejected candidates with
    reasons (build spec item 2), for callers that log rejects via the
    same mechanism ingest uses (`library.append_reject`)."""
    from sculptor.reference import validate_clip

    errors = validate_clip(clip)
    if errors:
        raise ValueError(
            "refusing to segment invalid reference clip:\n  - "
            + "\n  - ".join(errors))
    z = clip["root_pos_z"]
    fps = float(clip["fps"])
    result = segment_by_root_z_full(z, fps)
    out: list[dict] = []
    n = z.shape[0]
    for seg in result.segments:
        sliced: dict = {}
        for key, value in clip.items():
            if key in ("fps", "joint_names", "meta"):
                sliced[key] = value
                continue
            if isinstance(value, np.ndarray) and value.shape[:1] == (n,):
                sliced[key] = value[seg.start:seg.end]
            else:
                sliced[key] = value
        sliced["meta"] = dict(clip.get("meta") or {})
        sliced["_segment_frame_range"] = [seg.start, seg.end]
        out.append(sliced)
    return ClipSegmentationResult(segments=out, rejected=result.rejected)
