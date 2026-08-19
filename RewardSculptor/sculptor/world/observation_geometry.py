"""Pure geometry helpers shared by world compilation and admission."""

from __future__ import annotations

import math
from typing import Any, Sequence


HEIGHT_SCAN_GRID_SIZE_M = (1.6, 1.0)
HEIGHT_SCAN_GRID_RESOLUTION_M = 0.2


def inclusive_grid_sample_count(
    size: Sequence[Any], resolution: Any,
) -> int | None:
    """Return the sample count for a two-dimensional inclusive grid.

    MjLab ``GridPatternCfg`` samples both endpoints of each extent, so an
    extent containing ``n`` resolution intervals produces ``n + 1`` rays.
    Invalid or non-integral grids fail closed instead of being rounded into a
    different sensor contract.
    """
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in size
        )
        or not isinstance(resolution, (int, float))
        or isinstance(resolution, bool)
        or not math.isfinite(float(resolution))
        or float(resolution) <= 0.0
    ):
        return None

    counts: list[int] = []
    for extent in size:
        intervals = float(extent) / float(resolution)
        rounded = round(intervals)
        if abs(intervals - rounded) > 1e-6:
            return None
        counts.append(int(rounded) + 1)
    return math.prod(counts)


def height_scan_ray_count() -> int:
    """Return the ray count for the compiler's canonical height scan."""
    count = inclusive_grid_sample_count(
        HEIGHT_SCAN_GRID_SIZE_M,
        HEIGHT_SCAN_GRID_RESOLUTION_M,
    )
    if count is None:  # pragma: no cover - guarded by module constants
        raise RuntimeError("invalid canonical height-scan grid")
    return count


__all__ = [
    "HEIGHT_SCAN_GRID_RESOLUTION_M",
    "HEIGHT_SCAN_GRID_SIZE_M",
    "height_scan_ray_count",
    "inclusive_grid_sample_count",
]
