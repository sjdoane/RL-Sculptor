"""Dataset ingesters: LAFAN1-g1 (CSV) and fleaven-g1 (npy) -> validated
library clips (§R1_BUILD_SPEC decisions 1, 3, 5, 9).

Both source datasets share one column layout, VERIFIED against the real
data (not re-derived here — see R1_BUILD_SPEC "Verified ground truth"):
36 columns per frame = root position xyz [0:3] + root orientation
quaternion **xyzw** [3:7] + 29 joint angles (radians) [7:36], in the
`G1_29` order from `sculptor.eval.robot_manifest`. Ingest converts the
quaternion to wxyz (MuJoCo/clip convention, §decision 1) and verifies —
never assumes — the joint order via `assert_name_axis_contract` +
`resolve_joint_roles`, HARD-FAILING a clip on resolution failure rather
than silently reordering (§Hard rules).

Downloads happen only when a CLI command actually runs `ingest_source`;
nothing in this module is imported or called by the test suite in a way
that reaches the network (`_http_get_bytes` / `_hf_list_files` /
`_hf_list_files_all_pages` are the only network call sites, all leaves
called solely from `ingest_source` / `enumerate_fleaven_g1_all`).

Full-tree fleaven-g1 enumeration (2026-07-09): the HF tree API
(`.../tree/main/<path>?recursive=true`) is PAGINATED — 1000 entries per
page, next-page URL in the `Link: <url>; rel="next"` response header
(RFC 5988 style, verified empirically against the live dataset: 19
pages, 17769 files, ~6.95 GB under `g1/`). `_hf_list_files` (single
page, used by the default/slice path) is UNCHANGED and stays
byte-identical for backward compatibility — `--all` routes through the
new `_hf_list_files_all_pages` walker instead.
"""
from __future__ import annotations

import csv
import io
import json
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from sculptor.eval.joint_resolver import (
    assert_name_axis_contract,
    resolve_joint_roles,
)
from sculptor.eval.robot_manifest import G1_29
from sculptor.refs import library
from sculptor.reference import validate_clip

N_COLS = 36
ROOT_POS_SLICE = slice(0, 3)
ROOT_QUAT_XYZW_SLICE = slice(3, 7)
JOINT_SLICE = slice(7, 36)
N_JOINTS = 29

#: §Hard rule: "the datasets' documented order matches G1_29 — assert this
#: assumption explicitly ... so a future dataset change fails loudly."
#: This tuple is the README-documented G1 order for BOTH source datasets,
#: hardcoded here (not re-imported from G1_29) so a future edit to
#: robot_manifest.G1_29 that silently changes order is also caught —
#: the assertion below compares two independently-written tuples.
_DATASET_DOCUMENTED_G1_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


class DatasetFormatError(ValueError):
    """Reality contradicted the spec's verified dataset format (§Hard
    rules: STOP that item and report — never improvise a schema
    change). Raised instead of silently coping."""


class ClipRejected(ValueError):
    """A clip failed a QC gate (§decision 5). Caught by the batch
    ingester and logged to `index_rejects.jsonl`; never aborts the
    batch."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _assert_documented_order_matches_g1_29() -> None:
    """§Hard rule: fail loudly, at import time, if a future edit makes
    the hardcoded dataset-documented order and `robot_manifest.G1_29`
    diverge — a silent reorder is exactly the bug this guards against."""
    if _DATASET_DOCUMENTED_G1_ORDER != tuple(G1_29):
        raise DatasetFormatError(
            "dataset-documented G1 joint order no longer matches "
            "robot_manifest.G1_29 — DO NOT reorder silently; update the "
            "ingester's index-mapping table per R1_BUILD_SPEC decision 5"
        )


_assert_documented_order_matches_g1_29()


# ── quaternion convention (§decision 1) ─────────────────────────────────
def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert an (T, 4) quaternion array from xyzw (dataset/scipy
    convention) to wxyz (MuJoCo / clip convention). Pure reindex, no
    renormalization (callers validate unit-norm separately)."""
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
        raise ValueError(f"expected (T, 4) xyzw array, got shape {quat_xyzw.shape}")
    x, y, z, w = (quat_xyzw[:, i] for i in range(4))
    return np.stack([w, x, y, z], axis=1)


# ── label tokenization (shared with QC + index `labels`) ────────────────
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])")


def tokenize_label(stem: str) -> list[str]:
    """Split a source clip stem into lowercase tokens: camelCase and
    digit boundaries split, `_`/`-` split, empties dropped. E.g.
    `"fallAndGetUp1_subject1"` -> `["fall","and","get","up","1",
    "subject","1"]`."""
    s = _CAMEL_BOUNDARY.sub("_", str(stem))
    s = re.sub(r"[_\-]+", "_", s)
    parts = [p for p in s.split("_") if p]
    return [p.lower() for p in parts]


# ── QC gates (§decision 5) ───────────────────────────────────────────────
_MAX_JOINT_ANGLE_RAD = 2.0 * np.pi
_MAX_JOINT_DELTA_RAD_AT_30FPS = 0.5

#: §R1_BUILD_SPEC W2 item 0 (orchestrator ruling on W1's escalation):
#: real fleaven floor-contact clips have `root_pos_z` dipping slightly
#: negative (observed min -3.85 mm) — retargeting float noise, not a
#: real below-ground pose. `sculptor.reference.validate_clip` requires
#: `root_pos_z` strictly positive (that invariant is NOT touched here),
#: so a clip in this noise band would otherwise hard-reject on schema
#: validation despite being a legitimate floor-contact motion. Anything
#: past this floor is NOT noise — reject it (§decision 5's root-z gate
#: still owns the general (0, 2.5) m bound; this clamp only covers the
#: narrow noise band immediately below zero).
_ROOT_Z_CLAMP_NOISE_FLOOR_M = -0.05
_ROOT_Z_CLAMP_TARGET_M = 1e-4


def _clamp_root_z_noise(root_pos: np.ndarray) -> Optional[dict[str, Any]]:
    """Clamp `root_pos[:, 2]` in place when its minimum is within the
    retargeting-noise band `[-0.05, 0]` m, returning a qc-block entry
    describing what happened (or None if no clamp was needed). Does NOT
    reject — a min below -0.05 m is handled by the caller as a hard
    reject (`root_z_below_ground`), separate from this clamp path."""
    z = root_pos[:, 2]
    z_min = float(z.min())
    if not (_ROOT_Z_CLAMP_NOISE_FLOOR_M <= z_min <= 0.0):
        return None
    mask = z < _ROOT_Z_CLAMP_TARGET_M
    n_frames = int(mask.sum())
    root_pos[:, 2] = np.maximum(z, _ROOT_Z_CLAMP_TARGET_M)
    return {"n_frames": n_frames, "min_before": round(z_min, 6)}


def _plausibility_checks(
    root_pos: np.ndarray, joint_pos: np.ndarray, fps: float,
) -> tuple[list[str], list[str]]:
    """Returns (hard_fail_reasons, flagged_notes). Flagged notes are
    recorded in qc but do NOT reject the clip (§decision 5: "flag,
    don't fail")."""
    fails: list[str] = []
    flags: list[str] = []
    if np.abs(joint_pos).max() > _MAX_JOINT_ANGLE_RAD:
        fails.append(
            f"joint angle magnitude exceeds 2*pi: max={np.abs(joint_pos).max():.3f}")
    z = root_pos[:, 2]
    if not ((z > 0).all() and (z < 2.5).all()):
        fails.append(f"root z outside (0, 2.5) m: range={[float(z.min()), float(z.max())]}")
    if joint_pos.shape[0] >= 2:
        # Per-frame delta normalized to a 30fps-equivalent step so the
        # 0.5 rad budget is comparable across the datasets' different fps.
        deltas = np.abs(np.diff(joint_pos, axis=0)) * (30.0 / max(fps, 1e-6))
        max_delta = float(deltas.max())
        if max_delta > _MAX_JOINT_DELTA_RAD_AT_30FPS:
            flags.append(
                f"per-frame joint delta exceeds 0.5 rad@30fps-equivalent: "
                f"max={max_delta:.3f}")
    return fails, flags


#: Motion-class label -> content check (§decision 5). Keyed by any token
#: in the tokenized stem. Each check receives (root_z, horiz_disp_m) and
#: returns a reject reason or None.
def _check_fall_getup(z: np.ndarray, horiz_disp_m: float) -> Optional[str]:
    if not (z.min() < 0.35 and z.max() > 0.6):
        return (
            "fall/getup clip must have min(root_z) < 0.35 AND "
            f"max(root_z) > 0.6; got [{z.min():.3f}, {z.max():.3f}]")
    return None


def _check_locomotion(z: np.ndarray, horiz_disp_m: float) -> Optional[str]:
    if not (horiz_disp_m > 1.0):
        return (
            "walk/run/sprint clip must have horizontal displacement > 1 m; "
            f"got {horiz_disp_m:.3f} m")
    return None


#: token -> (motion class name, check fn). First matching token wins.
_MOTION_CLASS_CHECKS: dict[str, tuple[str, Callable[[np.ndarray, float], Optional[str]]]] = {
    "fall": ("fall/getup", _check_fall_getup),
    "getup": ("fall/getup", _check_fall_getup),
    "walk": ("locomotion", _check_locomotion),
    "run": ("locomotion", _check_locomotion),
    "sprint": ("locomotion", _check_locomotion),
}


def _motion_class_check(
    tokens: list[str], root_pos: np.ndarray,
) -> Optional[str]:
    z = root_pos[:, 2]
    horiz = root_pos[:, :2]
    horiz_disp_m = float(np.linalg.norm(horiz[-1] - horiz[0]))
    for tok in tokens:
        hit = _MOTION_CLASS_CHECKS.get(tok)
        if hit is not None:
            _, check_fn = hit
            return check_fn(z, horiz_disp_m)
    return None


def _run_qc_gates(
    *, root_pos: np.ndarray, quat_wxyz: np.ndarray, joint_pos: np.ndarray,
    fps: float, tokens: list[str],
) -> dict[str, Any]:
    """Runs every §decision-5 gate. Raises `ClipRejected` on the first
    hard failure; returns a `qc` dict (for provenance) otherwise, with
    `checks` listing what ran and any non-fatal flags."""
    n = root_pos.shape[0]
    checks: list[str] = ["column_count", "finite", "min_frames"]
    if n < 30:
        raise ClipRejected(f"T={n} frames < minimum 30")
    all_finite = (
        np.isfinite(root_pos).all() and np.isfinite(quat_wxyz).all()
        and np.isfinite(joint_pos).all())
    if not all_finite:
        raise ClipRejected("non-finite values in root pos/quat/joint arrays")

    # §R1_BUILD_SPEC W2 item 0: clamp retargeting float noise BEFORE the
    # (0, 2.5) m plausibility gate and BEFORE `validate_clip`'s strictly-
    # positive check would otherwise hard-reject a legitimate
    # floor-contact clip. A min below the -0.05 m noise floor is a real
    # below-ground pose, not noise — hard-reject it explicitly (distinct
    # reason from the general root-z-out-of-range plausibility fail).
    z0 = float(root_pos[:, 2].min())
    if z0 < _ROOT_Z_CLAMP_NOISE_FLOOR_M:
        raise ClipRejected(
            f"root_z_below_ground: min={z0:.6f} < "
            f"{_ROOT_Z_CLAMP_NOISE_FLOOR_M} m noise floor")
    root_z_clamped = _clamp_root_z_noise(root_pos)
    if root_z_clamped is not None:
        checks.append("root_z_clamp")

    checks.append("joint_mapping")
    # 1) Column-count contract: the sliced joint block must be exactly as
    #    wide as G1_29 — HARD-FAILS (raises) on any width mismatch, which
    #    is the only thing the raw numeric data can attest to (a CSV/npy
    #    row carries no joint names, so per-column identity is NOT
    #    independently checkable from the data alone; that assumption is
    #    the `_assert_documented_order_matches_g1_29` module-load guard
    #    above plus the README cross-check performed once by the human
    #    who verified R1_BUILD_SPEC's "Verified ground truth" section).
    assert_name_axis_contract(list(G1_29), joint_pos.shape[1])
    # 2) Manifest sanity: G1_29's own names must resolve to a clean 1:1
    #    permutation under `resolve_joint_roles` (no duplicate/ambiguous
    #    joint names) — catches a corrupted/edited manifest before it's
    #    trusted as `joint_names` on every clip we persist.
    res = resolve_joint_roles(list(G1_29), list(G1_29))
    if not res.ok or sorted(res.resolved.values()) != list(range(N_JOINTS)):
        raise ClipRejected(
            "G1_29 manifest sanity check failed (ambiguous/duplicate "
            "joint names) — refusing to trust joint_names for this "
            "ingest: " + "; ".join(res.problems()))

    checks.append("plausibility")
    fails, flags = _plausibility_checks(root_pos, joint_pos, fps)
    if fails:
        raise ClipRejected("; ".join(fails))

    checks.append("motion_class")
    mc_reason = _motion_class_check(tokens, root_pos)
    if mc_reason is not None:
        raise ClipRejected(mc_reason)

    z = root_pos[:, 2]
    qc_out: dict[str, Any] = {
        "duration_s": round(n / float(fps), 4),
        "root_z_range": [round(float(z.min()), 4), round(float(z.max()), 4)],
        "n_frames": n,
        "checks": checks,
        "flags": flags,
    }
    if root_z_clamped is not None:
        qc_out["root_z_clamped"] = root_z_clamped
    return qc_out


# ── shared: raw (T, 36) array -> validated library clip ─────────────────
def _rows_to_clip(
    rows: np.ndarray, *, fps: float, tokens: list[str],
) -> tuple[dict, dict]:
    """Common path for both dataset ingesters once they've produced a
    (T, 36) float array. Returns (clip_dict, qc_dict); raises
    `DatasetFormatError` on structural mismatch, `ClipRejected` on a QC
    gate failure."""
    if rows.ndim != 2 or rows.shape[1] != N_COLS:
        raise DatasetFormatError(
            f"expected (T, {N_COLS}) rows, got shape {rows.shape}")
    root_pos = rows[:, ROOT_POS_SLICE].astype(np.float64)
    quat_xyzw = rows[:, ROOT_QUAT_XYZW_SLICE].astype(np.float64)
    joint_pos = rows[:, JOINT_SLICE].astype(np.float64)
    quat_wxyz = xyzw_to_wxyz(quat_xyzw)

    qc = _run_qc_gates(
        root_pos=root_pos, quat_wxyz=quat_wxyz, joint_pos=joint_pos,
        fps=fps, tokens=tokens)

    clip = {
        "root_pos_z": root_pos[:, 2].copy(),
        "root_pos_xy": root_pos[:, :2].copy(),
        "root_quat_wxyz": quat_wxyz,
        "joint_pos": joint_pos,
        "joint_names": list(G1_29),
        "fps": float(fps),
        "meta": {"source": "dataset", "tokens": tokens},
    }
    errors = validate_clip(clip)
    if errors:
        raise ClipRejected("clip schema validation failed: " + "; ".join(errors))
    return clip, qc


# ── LAFAN1-g1 CSV ─────────────────────────────────────────────────────────
LAFAN1_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
LAFAN1_FPS = 30.0


def parse_lafan1_csv(raw: bytes, *, stem: str, fps: float = LAFAN1_FPS) -> tuple[dict, dict]:
    """LAFAN1-g1: headerless CSV, 36 float columns/row, 30 fps."""
    text = raw.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    data: list[list[float]] = []
    for row in reader:
        if not row:
            continue
        try:
            data.append([float(x) for x in row])
        except ValueError as e:
            raise DatasetFormatError(f"non-numeric CSV row in {stem}: {e}") from e
    if not data:
        raise DatasetFormatError(f"empty CSV: {stem}")
    widths = {len(r) for r in data}
    if widths != {N_COLS}:
        raise DatasetFormatError(
            f"{stem}: expected every row to have {N_COLS} columns, saw widths {widths}")
    rows = np.asarray(data, dtype=np.float64)
    tokens = tokenize_label(stem)
    return _rows_to_clip(rows, fps=fps, tokens=tokens)


# ── fleaven-g1 npy ────────────────────────────────────────────────────────
FLEAVEN_REPO = "fleaven/Retargeted_AMASS_for_robotics"
_FLEAVEN_FPS_RE = re.compile(r"_poses_(\d+)_jpos$")

#: SMPL-X "stageii" fits (GRAB, CNRS, SOMA, WEIZMANN, MOYO_smplh_gendered —
#: 3929 clips, verified 2026-07-11 against index_rejects.jsonl) name files
#: `..._stageii_{fps}_jpos.npy` instead of `..._poses_{fps}_jpos.npy`: the
#: fps digits ARE present, just not preceded by the literal "poses" token
#: the original regex required.
#:
#: Authoritative source: the dataset's OWN reader, `g1/visualize.py`
#: (hosted in this same HF repo —
#: https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics/
#: blob/main/g1/visualize.py), function `read_rtj()`:
#:     #get frame rate from file name
#:     fr = fpath[-12:-9]
#:     fr = int(fr[1:]) if fr[0]=='_' else int(fr)
#: i.e. the author's own convention is "fps = the digits immediately
#: preceding '_jpos' in the filename", independent of the marker word
#: before it. Cross-checked against the full-tree manifest
#: (`.fleaven_g1_all_manifest.json`, 17717 files, generated 2026-07-09):
#: exactly two marker words appear anywhere in the repo — "poses" (13788
#: files) and "stageii" (3929 files) — and EVERY file matches a generic
#: `_(\d+)_jpos$` pattern (0 unmatched). Within "stageii", the recovered
#: fps token is constant per subset directory (GRAB=120, CNRS=120,
#: SOMA=120, WEIZMANN=120, MOYO_smplh_gendered=60 — one value per subset,
#: never mixed), consistent with a per-subset AMASS capture rate rather
#: than a per-clip/session id.
_FLEAVEN_FPS_STAGEII_RE = re.compile(r"_stageii_(\d+)_jpos$")
_FLEAVEN_VISUALIZE_PY_URL = (
    "https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics/"
    "blob/main/g1/visualize.py"
)


def _fleaven_fps_from_filename(stem: str) -> tuple[float, dict[str, Any]]:
    """Returns `(fps, fps_provenance)`. `fps_provenance` is a small dict
    recorded into the clip's `qc` (and thus its provenance.json) so a
    recovered fps is always traceable back to its source — never a bare
    unattributed number."""
    m = _FLEAVEN_FPS_RE.search(stem)
    if m:
        return float(m.group(1)), {
            "pattern": "poses", "recovered": False,
            "note": "fps encoded via the known-good "
                    "'..._poses_{fps}_jpos' filename pattern",
        }
    m = _FLEAVEN_FPS_STAGEII_RE.search(stem)
    if m:
        return float(m.group(1)), {
            "pattern": "stageii", "recovered": True,
            "source": _FLEAVEN_VISUALIZE_PY_URL,
            "note": (
                "fps recovered from the '..._stageii_{fps}_jpos' filename "
                "token per the dataset author's own g1/visualize.py "
                "read_rtj() convention: fps = digits immediately "
                "preceding '_jpos', independent of the marker word before "
                "it. Cross-checked against the full-tree manifest: fps is "
                "constant per subset directory."),
        }
    raise DatasetFormatError(
        f"fleaven filename does not encode fps via a recognized pattern "
        f"(expected '..._poses_{{fps}}_jpos' or "
        f"'..._stageii_{{fps}}_jpos', and no other marker word has been "
        f"verified against an authoritative fps source): {stem}")


def parse_fleaven_npy(raw: bytes, *, stem: str) -> tuple[dict, dict]:
    """fleaven-g1: one (T, 36) float64 .npy per clip; fps from filename.
    `qc["fps_provenance"]` always records which filename pattern supplied
    the fps and, for a recovered ("stageii") pattern, the authoritative
    source it was verified against — see `_fleaven_fps_from_filename`."""
    try:
        rows = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as e:  # noqa: BLE001 — surfaced as a format error
        raise DatasetFormatError(f"could not load npy {stem}: {e}") from e
    fps, fps_provenance = _fleaven_fps_from_filename(stem)
    tokens = tokenize_label(stem)
    clip, qc = _rows_to_clip(np.asarray(rows, dtype=np.float64), fps=fps, tokens=tokens)
    qc["fps_provenance"] = fps_provenance
    return clip, qc


# ── HTTP (network call sites — never reached by tests) ───────────────────
_HF_RESOLVE_FMT = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
_HF_TREE_API_FMT = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}?recursive=true"
_USER_AGENT = "Reward-Sculptor/0.1 (reference-library ingest)"


def _hf_resolve_url(repo: str, path: str) -> str:
    """Build a HF `resolve/main` download URL, percent-encoding the path
    (fleaven filenames contain spaces and ` - `, which the raw f-string
    build sent unescaped — `http.client` then rejects the request with
    "URL can't contain control characters"; verified against a real
    fleaven ACCAD filename during this build). `/` is preserved as a
    path separator; every other reserved/unsafe character is quoted."""
    quoted = urllib.parse.quote(path, safe="/")
    return _HF_RESOLVE_FMT.format(repo=repo, path=quoted)


def _hf_tree_api_url(repo: str, path: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return _HF_TREE_API_FMT.format(repo=repo, path=quoted)


def _http_get_bytes(url: str, *, timeout_s: float = 120.0) -> bytes:
    """Stream to a temp file, verify readable, return bytes. Mirrors
    `kg/ingest.py`'s `_download_pdf_with_timeout` pattern (bounded-time,
    partial-file-then-rename) but returns bytes directly since clips are
    small enough (LAFAN1's biggest CSV is a few MB)."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp, tmp_path.open("wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        data = tmp_path.read_bytes()
        if not data:
            raise DatasetFormatError(f"empty download: {url}")
        return data
    finally:
        tmp_path.unlink(missing_ok=True)


def _hf_list_files(repo: str, path: str, *, timeout_s: float = 30.0) -> list[str]:
    """List file paths under `path` in a HuggingFace dataset repo via the
    public (ungated) tree API. Returns repo-relative paths.

    NOTE: this only fetches the FIRST page (≤1000 entries) of the tree
    API — the API is paginated (see `_hf_list_files_all_pages`) and this
    function's single-page behavior is intentionally left unchanged for
    backward compatibility with the existing (non `--all`) ingest path."""
    url = _hf_tree_api_url(repo, path)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [
        entry["path"] for entry in payload
        if entry.get("type") == "file"
    ]


#: RFC 5988-style `Link` header, e.g.
#: `<https://.../tree/main/g1?...&cursor=XYZ>; rel="next"`. HF's tree API
#: emits exactly this shape (verified empirically 2026-07-09: `curl -D -`
#: against the live fleaven-g1 tree endpoint) — one `Link` header, one
#: `rel="next"` entry, absent entirely on the last page.
_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def _parse_link_next(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    m = _LINK_NEXT_RE.search(link_header)
    return m.group(1) if m else None


def _http_get_json_with_retries(
    url: str, *, timeout_s: float, max_attempts: int, retry_delay_s: float,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """GET `url`, parse the JSON array body, and return `(payload,
    next_url)`. Transient failures (`URLError`/`OSError`/bad JSON) are
    retried up to `max_attempts` times with a flat `retry_delay_s` pause;
    after the last attempt fails this returns `(None, None)` and logs via
    `progress` rather than raising — callers skip-and-log the page
    (§task: "transient HTTP failures retried a few times, then
    skip-and-log"), they do not abort the whole enumeration."""
    log = progress or (lambda _msg: None)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
                next_url = _parse_link_next(resp.headers.get("Link"))
            payload = json.loads(body)
            return payload, next_url
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
            last_err = e
            log(f"[refs enumerate] page fetch failed (attempt {attempt}/"
                f"{max_attempts}) for {url}: {type(e).__name__}: {e}")
            if attempt < max_attempts:
                time.sleep(retry_delay_s)
    log(f"[refs enumerate] giving up on page after {max_attempts} attempts: "
        f"{url} ({type(last_err).__name__ if last_err else 'unknown'}: {last_err})")
    return None, None


def _hf_list_files_all_pages(
    repo: str, path: str, *, timeout_s: float = 30.0, max_attempts: int = 3,
    retry_delay_s: float = 1.0, max_pages: int = 10_000,
    progress: Optional[Callable[[str], None]] = None,
) -> list[dict[str, Any]]:
    """Walk EVERY page of the HF tree API's recursive listing under
    `path`, following the `Link: rel="next"` cursor until it's absent.
    Returns file entries only (`type == "file"`), each as
    `{"path": ..., "size": ...}`. A page that fails after retries is
    skipped (logged via `progress`) rather than aborting the whole walk
    — an incomplete listing is preferable to no listing (§task: "skip-
    and-log"); callers that need a complete listing should check log
    output / reconcile against the manifest.

    `max_pages` is a sanity backstop (repo has 19 pages today at 1000
    entries/page) — not expected to bind in practice."""
    log = progress or (lambda _msg: None)
    url: Optional[str] = _hf_tree_api_url(repo, path)
    out: list[dict[str, Any]] = []
    page = 0
    while url is not None and page < max_pages:
        page += 1
        payload, next_url = _http_get_json_with_retries(
            url, timeout_s=timeout_s, max_attempts=max_attempts,
            retry_delay_s=retry_delay_s, progress=progress)
        if payload is None:
            # Page permanently failed after retries. We cannot know its
            # `next` cursor (the failed request never returned one), so
            # the walk must stop here — everything after this page is
            # unreachable. Log loudly and return what we have so far.
            log(f"[refs enumerate] page {page} unrecoverable — stopping "
                f"walk early; {len(out)} file(s) enumerated before the gap")
            break
        for entry in payload:
            if entry.get("type") == "file":
                out.append({
                    "path": entry["path"],
                    "size": int(entry.get("size", 0)),
                })
        log(f"[refs enumerate] page {page}: +{sum(1 for e in payload if e.get('type') == 'file')} "
            f"file(s), {len(out)} total so far")
        url = next_url
    return out


def enumerate_fleaven_g1_all(
    *, timeout_s: float = 30.0, max_attempts: int = 3, retry_delay_s: float = 1.0,
    progress: Optional[Callable[[str], None]] = None,
) -> list[dict[str, Any]]:
    """Full-tree enumeration of every file under `g1/` in the fleaven-g1
    HF dataset repo, across ALL pages (not just the first 1000 entries).
    Returns `[{"path": ..., "size": ...}, ...]` filtered to `.npy` files
    (the repo also carries a handful of `.txt`/`.bib`/`.md`/`.py`
    metadata files scattered in the tree — verified empirically
    2026-07-09 — which are not motion clips and are excluded here so the
    manifest only ever lists ingestible files)."""
    entries = _hf_list_files_all_pages(
        FLEAVEN_REPO, "g1", timeout_s=timeout_s, max_attempts=max_attempts,
        retry_delay_s=retry_delay_s, progress=progress)
    return [e for e in entries if e["path"].endswith(".npy")]


# ── full-tree manifest (audit / resume) ──────────────────────────────────
_MANIFEST_MAX_AGE_S = 24 * 60 * 60  # 1 day (§task: "fresh (< 1 day)")


def write_fleaven_manifest(
    manifest_path: Path, entries: list[dict[str, Any]], *, source: str = "fleaven-g1",
) -> Path:
    """Write the enumerated file list as JSON: `{"source", "generated_at",
    "n_files", "total_bytes", "files": [{"path","size"}, ...]}`. Written
    atomically (temp file + rename) so a crash mid-write never leaves a
    corrupt manifest that `read_fleaven_manifest` would fail to parse."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "generated_at": _utc_now_iso(),
        "n_files": len(entries),
        "total_bytes": sum(int(e.get("size", 0)) for e in entries),
        "files": entries,
    }
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)
    return manifest_path


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def manifest_is_fresh(manifest_path: Path, *, max_age_s: float = _MANIFEST_MAX_AGE_S) -> bool:
    """True iff `manifest_path` exists, parses, and its `generated_at`
    timestamp is within `max_age_s` of now (§task: "fresh (< 1 day)").
    Any read/parse error is treated as NOT fresh (fail open to
    re-enumerating rather than trusting a corrupt manifest)."""
    if not manifest_path.is_file():
        return False
    try:
        from datetime import datetime

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        generated_at = datetime.fromisoformat(payload["generated_at"])
        now = datetime.fromisoformat(_utc_now_iso())
        age_s = (now - generated_at).total_seconds()
        return 0 <= age_s <= max_age_s
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return False


def read_fleaven_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload["files"])


def load_or_build_fleaven_manifest(
    manifest_path: Optional[Path], *, refresh: bool = False,
    timeout_s: float = 30.0, max_attempts: int = 3, retry_delay_s: float = 1.0,
    progress: Optional[Callable[[str], None]] = None,
) -> list[dict[str, Any]]:
    """Resolve the full `g1/**/*.npy` file list for `--all` ingest: reuse
    an existing fresh manifest unless `refresh` is set or none exists /
    it's stale, otherwise enumerate for real and (if `manifest_path` is
    given) persist the result for next time (§task item 3)."""
    log = progress or (lambda _msg: None)
    if manifest_path is not None and not refresh and manifest_is_fresh(manifest_path):
        entries = read_fleaven_manifest(manifest_path)
        log(f"[refs ingest] reusing fresh manifest {manifest_path} "
            f"({len(entries)} files)")
        return entries

    log("[refs ingest] enumerating full fleaven-g1 tree (this walks every "
        "page of the HF tree API — may take a while)")
    entries = enumerate_fleaven_g1_all(
        timeout_s=timeout_s, max_attempts=max_attempts,
        retry_delay_s=retry_delay_s, progress=progress)
    total_bytes = sum(int(e.get("size", 0)) for e in entries)
    log(f"[refs ingest] enumerated {len(entries)} file(s), "
        f"{total_bytes} bytes ({total_bytes / 1e9:.2f} GB)")
    if manifest_path is not None:
        write_fleaven_manifest(manifest_path, entries)
        log(f"[refs ingest] manifest written: {manifest_path}")
    return entries


@dataclass
class IngestSummary:
    accepted: list[str]
    rejected: list[tuple[str, str]]
    skipped_existing: list[str]


_SOURCES = ("lafan1-g1", "fleaven-g1")

#: §Problem 2 (2026-07-11): both current `ingest_source` datasets are
#: INHERENTLY g1-only — their raw 36-col rows are hardwired to the
#: `G1_29` joint order (`_run_qc_gates` asserts this at parse time via
#: `assert_name_axis_contract`/`resolve_joint_roles`, unconditionally,
#: regardless of the `robot` argument), and their source names say so
#: explicitly (`"lafan1-g1"`, `"fleaven-g1"`). This map makes that fact
#: machine-checked rather than merely implied by naming convention —
#: `ingest_source` refuses to register g1-schema data under a different
#: robot slug (which would silently mislabel it: a T1 clip whose actual
#: numbers are G1 joint angles). Other robots (e.g. `t1`) are ingested
#: through an entirely separate, already robot-generic pipeline —
#: `sculptor.refs.retarget.retarget_and_register` — which reads
#: `joint_names` from GMR's own output rather than assuming G1_29.
_SOURCE_ROBOT: dict[str, str] = {"lafan1-g1": "g1", "fleaven-g1": "g1"}


def ingest_source(
    source: str,
    *,
    filter_glob: Optional[str] = None,
    limit: Optional[int] = None,
    no_preview: bool = False,
    root: Optional[Path] = None,
    robot: str = "g1",
    full_tree: bool = False,
    manifest_path: Optional[Path] = None,
    refresh_manifest: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> IngestSummary:
    """Download + ingest a batch from one dataset source (§decision 9).
    The single network-reaching entry point workers/CLI should call.
    `no_preview=False` (default) attempts a best-effort `preview.png`
    render per accepted clip/segment via `sculptor.refs.preview`
    (§decision 8) — never blocks or fails ingest; `no_preview=True`
    skips preview generation entirely (also the automatic fallback when
    the preview module isn't importable or GL/EGL is unavailable).

    `full_tree=False` (default) is BYTE-IDENTICAL to the pre-existing
    behavior: a single-page `_hf_list_files` listing. `full_tree=True`
    (fleaven-g1 only) instead walks every page of the HF tree API via
    `load_or_build_fleaven_manifest` — the complete `g1/**/*.npy` file
    list — optionally reusing/writing a manifest at `manifest_path`.
    `filter_glob`/`limit` apply identically on top of either file list."""
    if source not in _SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {_SOURCES}")
    expected_robot = _SOURCE_ROBOT[source]
    if robot != expected_robot:
        raise ValueError(
            f"source {source!r} is inherently robot={expected_robot!r} "
            f"(its raw rows are hardwired to that robot's joint order — "
            f"see `_SOURCE_ROBOT`'s docstring); refusing to register it "
            f"under robot={robot!r}, which would silently mislabel "
            f"{expected_robot}-schema data as a different robot")
    log = progress or (lambda _msg: None)
    repo = LAFAN1_REPO if source == "lafan1-g1" else FLEAVEN_REPO

    if full_tree:
        if source != "fleaven-g1":
            raise ValueError(
                f"full_tree=True is only supported for source='fleaven-g1' "
                f"(got {source!r})")
        entries = load_or_build_fleaven_manifest(
            manifest_path, refresh=refresh_manifest, progress=progress)
        files = [e["path"] for e in entries]
    else:
        listing_path = "g1"
        files = _hf_list_files(repo, listing_path)
        if source == "lafan1-g1":
            files = [f for f in files if f.endswith(".csv")]
        else:
            files = [f for f in files if f.endswith(".npy")]
    if filter_glob:
        import fnmatch

        files = [f for f in files if fnmatch.fnmatch(Path(f).name, filter_glob)]
    files.sort()
    if limit is not None:
        files = files[:limit]

    existing_hashes = library.indexed_content_hashes(root=root)
    summary = IngestSummary(accepted=[], rejected=[], skipped_existing=[])

    for rel_path in files:
        stem = Path(rel_path).stem
        url = _hf_resolve_url(repo, rel_path)
        log(f"[refs ingest] fetching {url}")
        try:
            raw = _http_get_bytes(url)
        except (urllib.error.URLError, OSError, DatasetFormatError) as e:
            summary.rejected.append((stem, f"download failed: {e}"))
            library.append_reject(
                "download_failed", {"clip_id": stem, "source_path": rel_path, "detail": str(e)},
                root=root)
            continue

        sha = library.content_sha256(raw)
        if sha in existing_hashes:
            summary.skipped_existing.append(library.slugify(stem))
            continue

        try:
            result = ingest_clip_bytes(
                raw, source=source, repo=repo, rel_path=rel_path, stem=stem,
                robot=robot, root=root, no_preview=no_preview, progress=log)
        except (DatasetFormatError, ClipRejected) as e:
            reason = e.reason if isinstance(e, ClipRejected) else str(e)
            summary.rejected.append((stem, reason))
            library.append_reject(
                "qc_failed" if isinstance(e, ClipRejected) else "format_error",
                {"clip_id": stem, "source_path": rel_path, "detail": reason},
                root=root)
            continue
        existing_hashes.add(sha)
        summary.accepted.append(result.clip_id)
        log(f"[refs ingest] accepted {result.clip_id}")

    return summary


def _try_render_preview(
    clip: dict, robot: str, clip_id: str, *, root: Optional[Path],
    progress: Optional[Callable[[str], None]] = None,
) -> None:
    """Best-effort preview.png generation for one library clip. Never
    raises: `sculptor.refs.preview` may not be importable yet (an
    earlier worker stage) or GL/EGL may be unavailable in this
    environment (§decision 8) — either way this logs (via `progress`,
    if given) and returns, leaving the clip valid with no preview.png.

    §Problem 2 (2026-07-11): the MJCF is resolved BY `robot` via
    `preview.resolve_mjcf_for_robot` — this used to unconditionally
    default to the G1 MJCF regardless of which robot the clip actually
    belongs to (invisible for fleaven/lafan1 ingest, both g1-only
    sources, but wrong in general)."""
    log = progress or (lambda _msg: None)
    try:
        from sculptor.refs.preview import (
            PreviewUnavailable, render_preview_png, resolve_mjcf_for_robot)
    except ModuleNotFoundError:
        log(f"[refs ingest] preview module not available — skipping "
            f"preview for {clip_id}")
        return
    out_path = library.clip_dir(robot, clip_id, root=root) / library.PREVIEW_FILENAME
    try:
        mjcf_path = resolve_mjcf_for_robot(robot)
        render_preview_png(clip, out_path, mjcf_path=mjcf_path)
    except PreviewUnavailable as e:
        log(f"[refs ingest] preview unavailable for {clip_id}: {e}")
    except Exception as e:  # noqa: BLE001 — preview must never block ingest
        log(f"[refs ingest] preview failed for {clip_id}: "
            f"{type(e).__name__}: {e}")


def persist_segments(
    clip: dict, *, clip_id: str, robot: str, source: dict[str, Any],
    license_: str, attribution: str, fps_source: float, tokens: list[str],
    parent_qc: dict[str, Any], root: Optional[Path] = None,
    no_preview: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> list["library.LibraryClip"]:
    """Segment `clip` (§decision 6, 2026-07-09 settled-start fix) and
    persist each accepted segment as a derived library clip
    (`<clip_id>--segNN`), logging QC-rejected candidates via
    `library.append_reject`. Shared by `ingest_clip_bytes` (segmenting a
    freshly-ingested parent) and `resegment_clip` (re-segmenting an
    already-indexed parent with the current rules) so both paths persist
    segments identically. Returns the list of persisted `LibraryClip`s in
    segment order."""
    from sculptor.reference import save_clip
    from sculptor.refs.segment import segment_clip_full

    seg_result = segment_clip_full(clip)
    for rej in seg_result.rejected:
        library.append_reject(
            "segment_qc_failed",
            {"clip_id": f"{clip_id}--seg??", "parent_clip_id": clip_id,
             "frame_range": [rej.start, rej.end], "detail": rej.reason},
            root=root)

    results: list[library.LibraryClip] = []
    for i, seg_clip in enumerate(seg_result.segments):
        frame_range = seg_clip.pop("_segment_frame_range")
        # §decision 4: clip_id = slugified source name + literal `--segNN`
        # suffix. `clip_id` is already a valid slug and `-` is a legal
        # clip_id character, so the separator is appended directly rather
        # than run through `slugify` again (which would collapse `--`
        # into a single `_` and lose it).
        seg_clip_id = f"{clip_id}--seg{i:02d}"
        library.validate_clip_id(seg_clip_id)
        seg_qc = dict(parent_qc)
        z = seg_clip["root_pos_z"]
        seg_qc["duration_s"] = round(z.shape[0] / float(clip["fps"]), 4)
        seg_qc["root_z_range"] = [round(float(z.min()), 4), round(float(z.max()), 4)]
        seg_qc["n_frames"] = int(z.shape[0])
        seg_labels = tokens + ["segment"]
        seg_prov = library.make_provenance(
            clip_id=seg_clip_id, robot=robot,
            source=source, license=license_, attribution=attribution,
            content_sha256_=library.content_sha256(
                seg_clip["root_pos_z"].tobytes()),
            tier="K", fps_source=fps_source,
            parent_clip_id=clip_id, frame_range=list(frame_range),
            joint_mapping={"identity": True}, labels=seg_labels,
            text=" ".join(seg_labels), qc=seg_qc,
        )
        seg_d = library.clip_dir(robot, seg_clip_id, root=root)
        save_clip(seg_d / library.CLIP_FILENAME, seg_clip)
        seg_prov_path = library.write_provenance(robot, seg_clip_id, seg_prov, root=root)
        results.append(library.LibraryClip(
            robot=robot, clip_id=seg_clip_id,
            clip_path=seg_d / library.CLIP_FILENAME,
            provenance_path=seg_prov_path, provenance=seg_prov))
        if not no_preview:
            _try_render_preview(
                seg_clip, robot, seg_clip_id, root=root, progress=progress)
    return results


def ingest_clip_bytes(
    raw: bytes, *, source: str, repo: str, rel_path: str, stem: str,
    robot: str = "g1", root: Optional[Path] = None, no_preview: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> "library.LibraryClip":
    """Parse + QC + persist ONE clip's raw bytes (CSV or npy, dispatched
    by `source`). Also runs the §decision-6 segmentation for clips whose
    tokens match fall/getup, persisting each segment as a derived clip.
    Shared by `ingest_source` and tests (which call this directly on
    synthetic bytes — no network). `no_preview=False` (default) attempts
    a best-effort `preview.png` render for the clip and each of its
    segments via `_try_render_preview` — never blocks/fails ingest."""
    if source == "lafan1-g1":
        clip, qc = parse_lafan1_csv(raw, stem=stem)
        license_ = "CC BY-NC-ND 4.0"
        attribution = f"LAFAN1 Retargeting Dataset ({repo}), {rel_path}"
        fps_source = LAFAN1_FPS
    elif source == "fleaven-g1":
        clip, qc = parse_fleaven_npy(raw, stem=stem)
        license_ = "cc-by-4.0"
        attribution = f"Retargeted AMASS for robotics ({repo}), {rel_path}"
        fps_source = clip["fps"]
    else:
        raise ValueError(f"unknown source {source!r}")

    clip_id = library.slugify(stem)
    sha = library.content_sha256(raw)
    tokens = tokenize_label(stem)
    prov = library.make_provenance(
        clip_id=clip_id, robot=robot,
        source={"kind": "hf_dataset", "repo": repo, "path": rel_path,
                "url": _hf_resolve_url(repo, rel_path)},
        license=license_, attribution=attribution,
        content_sha256_=sha, tier="K", fps_source=fps_source,
        joint_mapping={"identity": True}, labels=tokens,
        text=" ".join(tokens), qc=qc,
    )

    from sculptor.reference import save_clip

    d = library.clip_dir(robot, clip_id, root=root)
    save_clip(d / library.CLIP_FILENAME, clip)
    prov_path = library.write_provenance(robot, clip_id, prov, root=root)
    result = library.LibraryClip(
        robot=robot, clip_id=clip_id, clip_path=d / library.CLIP_FILENAME,
        provenance_path=prov_path, provenance=prov)
    if not no_preview:
        _try_render_preview(clip, robot, clip_id, root=root, progress=progress)

    # §decision 6: segment fall/getup clips at ingest time. QC-rejected
    # candidates (2026-07-09 fix) are logged to the same
    # `index_rejects.jsonl` mechanism as every other ingest rejection,
    # rather than silently vanishing — see `persist_segments`.
    if any(t in ("fall", "getup") for t in tokens):
        persist_segments(
            clip, clip_id=clip_id, robot=robot, source=prov["source"],
            license_=license_, attribution=attribution, fps_source=fps_source,
            tokens=tokens, parent_qc=qc, root=root, no_preview=no_preview,
            progress=progress)

    return result


class ResegmentError(ValueError):
    """Raised when `resegment_clip` can't find the requested parent clip,
    or the parent's provenance is missing the fields needed to rebuild
    segment provenance (§Hard rules: fail loudly, never improvise)."""


@dataclass
class ResegmentSummary:
    parent_clip_id: str
    robot: str
    removed: list[str]
    added: list[str]
    rejected: list[tuple[str, str]]  # (would-be clip_id, reason)
    dry_run: bool


def find_derived_segments(
    robot: str, parent_clip_id: str, *, root: Optional[Path] = None,
) -> list[str]:
    """Every indexed clip_id under `robot` whose provenance
    `parent_clip_id` matches `parent_clip_id`, scanned from disk (not the
    possibly-stale index) — mirrors `library.rebuild_index`'s own
    disk-scan approach. Sorted for determinism."""
    r = root or library.references_root()
    robot_d = library.robot_dir(robot, root=r)
    if not robot_d.is_dir():
        return []
    out: list[str] = []
    for clip_d in sorted(p for p in robot_d.iterdir() if p.is_dir()):
        prov_path = clip_d / library.PROVENANCE_FILENAME
        if not prov_path.is_file():
            continue
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if prov.get("parent_clip_id") == parent_clip_id:
            out.append(clip_d.name)
    return out


def resegment_clip(
    parent_clip_id: str, *, robot: str = "g1", root: Optional[Path] = None,
    dry_run: bool = False, no_preview: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> ResegmentSummary:
    """Re-run segmentation for an already-indexed parent clip using the
    CURRENT segmentation rules (`sculptor.refs.segment`), replacing its
    existing derived segments. Only clips whose provenance
    `parent_clip_id` matches `parent_clip_id` are touched — everything
    else in the library is untouched. `dry_run=True` computes and
    reports what would change without writing or deleting anything.
    Rebuilds `index.jsonl` after writing (skipped in dry-run)."""
    import shutil

    from sculptor.reference import load_clip

    log = progress or (lambda _msg: None)
    r = root or library.references_root()

    parent_dir = library.clip_dir(robot, parent_clip_id, root=r)
    parent_clip_path = parent_dir / library.CLIP_FILENAME
    parent_prov_path = parent_dir / library.PROVENANCE_FILENAME
    if not parent_clip_path.is_file() or not parent_prov_path.is_file():
        raise ResegmentError(
            f"no such parent clip in the library: robot={robot!r} "
            f"clip_id={parent_clip_id!r} (looked under {parent_dir})")

    parent_clip = load_clip(parent_clip_path)
    parent_prov = library.read_provenance(robot, parent_clip_id, root=r)
    for field_name in ("source", "license", "attribution", "fps_source", "labels"):
        if parent_prov.get(field_name) in (None, ""):
            raise ResegmentError(
                f"parent clip {parent_clip_id!r} provenance is missing "
                f"required field {field_name!r} — refusing to resegment")

    tokens = list(parent_prov["labels"])
    qc = dict(parent_prov.get("qc") or {})

    existing = find_derived_segments(robot, parent_clip_id, root=r)
    log(f"[refs resegment] parent={parent_clip_id} existing_segments={len(existing)}")

    # Compute the new segmentation BEFORE deleting anything, so a
    # dry-run (or a crash) never leaves the library without its old
    # segments and without new ones.
    from sculptor.refs.segment import segment_clip_full

    seg_result = segment_clip_full(parent_clip)
    would_add = [f"{parent_clip_id}--seg{i:02d}" for i in range(len(seg_result.segments))]
    rejected = [
        (f"{parent_clip_id}--seg??[{rej.start}:{rej.end}]", rej.reason)
        for rej in seg_result.rejected]

    if dry_run:
        log(f"[refs resegment] DRY RUN: would remove {len(existing)}, "
            f"add {len(would_add)}, reject {len(rejected)}")
        return ResegmentSummary(
            parent_clip_id=parent_clip_id, robot=robot, removed=existing,
            added=would_add, rejected=rejected, dry_run=True)

    for seg_clip_id in existing:
        shutil.rmtree(library.clip_dir(robot, seg_clip_id, root=r), ignore_errors=True)
        log(f"[refs resegment] removed {seg_clip_id}")

    persisted = persist_segments(
        parent_clip, clip_id=parent_clip_id, robot=robot,
        source=parent_prov["source"], license_=parent_prov["license"],
        attribution=parent_prov["attribution"],
        fps_source=parent_prov["fps_source"], tokens=tokens, parent_qc=qc,
        root=r, no_preview=no_preview, progress=progress)
    added = [lc.clip_id for lc in persisted]

    library.rebuild_index(root=r)
    log(f"[refs resegment] removed={len(existing)} added={len(added)} "
        f"rejected={len(rejected)}")
    return ResegmentSummary(
        parent_clip_id=parent_clip_id, robot=robot, removed=existing,
        added=added, rejected=rejected, dry_run=False)
