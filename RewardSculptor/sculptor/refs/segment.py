"""Deterministic root_z hysteresis segmentation for multi-repetition
clips (§R1_BUILD_SPEC decision 6).

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
interval, padded +/-0.5 s, with a 2 s minimum length (segments shorter
than that are dropped — they're noise, not a real fall+recover cycle).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STANDING_Z = 0.60
DOWN_Z = 0.35
STANDING_SUSTAIN_S = 1.0
DOWN_SUSTAIN_S = 0.5
PAD_S = 0.5
MIN_SEGMENT_S = 2.0


@dataclass(frozen=True)
class Segment:
    """A segmentation result in frame indices, half-open `[start, end)`."""

    start: int
    end: int

    @property
    def n_frames(self) -> int:
        return self.end - self.start


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


def segment_by_root_z(z: np.ndarray, fps: float) -> list[Segment]:
    """Segment a root-height trace into down->recover cycles.

    Deterministic, offline, no LLM. Returns segments in frame order;
    overlapping/adjacent padded windows are NOT merged (§decision 6 says
    nothing about merging, and keeping cycles distinct is what a
    multi-rep RSI consumer wants — each is a complete fall+recover).
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or z.size == 0:
        return []
    fps = float(fps)
    standing_min = max(1, int(round(STANDING_SUSTAIN_S * fps)))
    down_min = max(1, int(round(DOWN_SUSTAIN_S * fps)))
    pad = int(round(PAD_S * fps))
    min_len = max(1, int(round(MIN_SEGMENT_S * fps)))

    standing_runs = _sustained_runs(z > STANDING_Z, standing_min)
    down_runs = _sustained_runs(z < DOWN_Z, down_min)
    if not down_runs or not standing_runs:
        return []

    segments: list[Segment] = []
    for d_start, d_end in down_runs:
        # Next sustained-standing run that starts at/after this down run
        # ends — "down-interval -> next sustained-standing".
        nxt = next(
            (s for s in standing_runs if s[0] >= d_end), None)
        if nxt is None:
            continue
        seg_start = max(0, d_start - pad)
        seg_end = min(z.size, nxt[1] + pad)
        if (seg_end - seg_start) < min_len:
            continue
        segments.append(Segment(seg_start, seg_end))
    return segments


def segment_clip(clip: dict) -> list[dict]:
    """Apply `segment_by_root_z` to a validated reference clip and slice
    every array-valued key (+meta preserved) into per-segment clip dicts.
    Callers are responsible for re-validating and persisting each slice
    with derived provenance (`parent_clip_id` / `frame_range`)."""
    from sculptor.reference import validate_clip

    errors = validate_clip(clip)
    if errors:
        raise ValueError(
            "refusing to segment invalid reference clip:\n  - "
            + "\n  - ".join(errors))
    z = clip["root_pos_z"]
    fps = float(clip["fps"])
    segments = segment_by_root_z(z, fps)
    out: list[dict] = []
    n = z.shape[0]
    for seg in segments:
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
    return out
