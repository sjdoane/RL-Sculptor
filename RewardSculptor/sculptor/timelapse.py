"""sculptor/timelapse.py — build the end-of-run report.

Two outputs, both driven from `<project>/runs/iter_*/` + `reports/` artifacts
produced by `sculpt run`:

  1. `final.mp4`: title card + side-by-side (or grid) of rollout videos
     sampled from iterations 1, N/2, N. Each panel labeled with its iter
     number and the configured primary_metric value. Composed via ffmpeg
     (bundled binary from imageio-ffmpeg used when system ffmpeg isn't
     on PATH); panel/title labels are pre-rendered to PNG via PIL so we
     sidestep ffmpeg drawtext's font-path escaping on Windows.

  2. `<project>/reports/final_report.md`:
       - Behavior goal + starting/ending behavior descriptions
         (from the first and last `rollout/behavior.json`).
       - Top 3 most impactful edits by primary_metric delta.
       - Literature map: per-term citations from `reports/provenance.json`.
       - Candidate novel contributions (every applied edit with
         paper_refs=[] whose rationale begins with "novel.").
       - Summary table (iter, primary_metric, num_references_added,
         num_novel_edits).
       - Pointer to the full CHANGELOG.md.

The module is useful standalone (`build_report(config_path, out_mp4)`)
and is wired into the CLI as `sculpt report`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_RECEIPT_SCHEMA = 1
REPORT_RECEIPT_NAME = "final_report.receipt.json"


# ── Small path helpers ──────────────────────────────────────────────────
def _parse_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py310 fallback
        import tomli as tomllib  # type: ignore[no-redef]
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_json(path: Path, default=None):
    if not path.is_file():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {} if default is None else default


def _load_reward_spec(path: Path) -> tuple[Path, dict]:
    """Return ``(path, REWARD_SPEC)`` without mutating import state."""
    spec = importlib.util.spec_from_file_location(
        f"_sculpt_final_reward_{abs(hash(str(path)))}", path)
    if spec is None or spec.loader is None:
        return path, {}
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        return path, {}
    return path, getattr(mod, "REWARD_SPEC", {}) or {}


def _load_final_reward_spec(rewards_dir: Path) -> tuple[Path, dict]:
    """Return ``(path_to_latest_vN.py, REWARD_SPEC)`` as a fallback."""
    best: tuple[int, Path] | None = None
    for p in rewards_dir.glob("v*.py"):
        m = re.fullmatch(r"v(\d+)", p.stem)
        if m:
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, p)
    if best is None:
        return rewards_dir / "v0.py", {}
    return _load_reward_spec(best[1])


def _find_iter_dirs(runs_dir: Path) -> list[Path]:
    dirs: list[tuple[int, Path]] = []
    if not runs_dir.is_dir():
        return []
    for d in runs_dir.iterdir():
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if m and d.is_dir() and not d.is_symlink():
            dirs.append((int(m.group(1)), d))
    dirs.sort(key=lambda x: x[0])
    return [d for _, d in dirs]


def _sha256_file(path: Path) -> str | None:
    """Hash one plain file, returning ``None`` for missing/linked inputs."""
    try:
        if path.is_symlink() or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _completion_receipt(iter_dir: Path) -> dict[str, Any] | None:
    """Verify the modern completion marker and exact phase/checkpoint bytes.

    Legacy rollout/fitness heuristics remain useful for recovery, but reports
    must not upgrade them into completed scientific iterations. Schema-3
    verification is delegated to the same CPU/data-only core contract used by
    the backend, including its request/input/completion manifest relationships.
    """
    try:
        from sculptor.run_manifests import verify_iteration_completion_marker

        return verify_iteration_completion_marker(iter_dir)
    except Exception:  # noqa: BLE001 - report completion fails closed
        return None


def _completed_iter_dirs(runs_dir: Path) -> list[Path]:
    return [
        iter_dir for iter_dir in _find_iter_dirs(runs_dir)
        if _completion_receipt(iter_dir) is not None
    ]


def _report_claim_inputs(
    selected_dir: Path,
    completion: dict[str, Any],
) -> dict[str, Any]:
    """Digest the fixed input set behind a selected-policy report claim."""
    checkpoint = selected_dir / str(completion["checkpoint"])
    artifact_tuple = _load_json(selected_dir / "artifact_tuple.json")
    reward_ref = ((artifact_tuple.get("refs") or {}).get("reward") or {})
    reward_path = reward_ref.get("path")
    declared_reward_sha = reward_ref.get("sha256")
    actual_reward_sha = None
    if isinstance(reward_path, str) and reward_path:
        project = selected_dir.parent.parent
        try:
            reward_candidate = (project / reward_path).resolve(strict=True)
            if (
                not (project / reward_path).is_symlink()
                and reward_candidate.is_relative_to(project.resolve(strict=True))
            ):
                actual_reward_sha = _sha256_file(reward_candidate)
        except (OSError, RuntimeError):
            actual_reward_sha = None
    return {
        "completion_marker_sha256": completion.get("marker_sha256"),
        "phase_manifests_sha256": completion.get("phase_manifests_sha256"),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "artifact_tuple_sha256": _sha256_file(
            selected_dir / "artifact_tuple.json"
        ),
        "selected_reward_path": reward_path,
        "selected_reward_declared_sha256": declared_reward_sha,
        "selected_reward_sha256": actual_reward_sha,
        "fitness_sha256": _sha256_file(selected_dir / "fitness.json"),
        "metrics_sha256": _sha256_file(selected_dir / "metrics.json"),
        "behavior_sha256": _sha256_file(
            selected_dir / "rollout" / "behavior.json"
        ),
        "rollout_sha256": _sha256_file(
            selected_dir / "rollout" / "rollout.mp4"
        ),
        "run_context_sha256": _sha256_file(
            selected_dir / "run_context.json"
        ),
        "physical_acceptance_contract_sha256": _sha256_file(
            selected_dir / "physical_acceptance_contract.json"
        ),
        "physical_acceptance_receipt_sha256": _sha256_file(
            selected_dir / "physical_acceptance_receipt.json"
        ),
    }


def _iter_number(iter_dir: Path) -> int:
    match = re.fullmatch(r"iter_(\d+)", iter_dir.name)
    return int(match.group(1)) if match else -1


def _select_report_iter_dir(
    project: Path,
    iter_dirs: list[Path],
    selection_authority: dict[str, Any] | None,
) -> Path | None:
    """Resolve a selected policy only from a byte-bound server authority.

    Score and recency are deliberately absent.  The backend may issue an
    authority only after the canonical selection receipt and independent
    objective proof pass.  Standalone/legacy callers still receive a useful
    descriptive run-history report, but it contains no ``Selected`` claim.
    """
    if not isinstance(selection_authority, dict):
        return None
    selected = selection_authority.get("selected_iter_index")
    selection_sha = selection_authority.get("selection_receipt_sha256")
    checkpoint_sha = selection_authority.get("selected_checkpoint_sha256")
    objective_sha = selection_authority.get("objective_evidence_sha256")
    objective_receipt = selection_authority.get("objective_evidence_receipt")
    claim_inputs = selection_authority.get("claim_inputs")
    claim_inputs_sha = selection_authority.get("claim_inputs_sha256")
    authority_status = selection_authority.get("status")
    authority_digest = selection_authority.get("authority_digest")
    unsigned = {
        key: value for key, value in selection_authority.items()
        if key != "authority_digest"
    }
    if (
        authority_status != "verified"
        or type(selected) is not int
        or not isinstance(selection_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", selection_sha) is None
        or not isinstance(checkpoint_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha) is None
        or not isinstance(objective_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", objective_sha) is None
        or not isinstance(objective_receipt, dict)
        or objective_receipt.get("objective_proof_status") != "passed"
        or objective_sha != _canonical_digest(objective_receipt)
        or not isinstance(claim_inputs, dict)
        or not isinstance(claim_inputs_sha, str)
        or claim_inputs_sha != _canonical_digest(claim_inputs)
        or not isinstance(authority_digest, str)
        or authority_digest != _canonical_digest(unsigned)
        or _sha256_file(project / "reports" / "selection.json") != selection_sha
    ):
        return None
    matches = [d for d in iter_dirs if _iter_number(d) == selected]
    if len(matches) != 1:
        return None
    selected_dir = matches[0]
    completion = _completion_receipt(selected_dir)
    if completion is None:
        return None
    current_claim_inputs = _report_claim_inputs(selected_dir, completion)
    if (
        claim_inputs != current_claim_inputs
        or checkpoint_sha != completion.get("checkpoint_sha256")
        or current_claim_inputs.get("selected_reward_declared_sha256")
        != current_claim_inputs.get("selected_reward_sha256")
    ):
        return None
    return selected_dir


def _load_selected_reward_spec(
    project: Path,
    selected_iter_dir: Path | None,
    rewards_dir: Path,
) -> tuple[Path, dict]:
    """Load the reward pinned to the selected policy's immutable tuple."""
    if selected_iter_dir is not None:
        artifact_tuple = _load_json(selected_iter_dir / "artifact_tuple.json")
        reward_ref = ((artifact_tuple.get("refs") or {}).get("reward") or {})
        reward_rel = reward_ref.get("path")
        reward_sha = reward_ref.get("sha256")
        if (
            isinstance(reward_rel, str)
            and reward_rel
            and isinstance(reward_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", reward_sha) is not None
        ):
            raw_candidate = project / reward_rel
            try:
                candidate = raw_candidate.resolve(strict=True)
                project_root = project.resolve(strict=True)
            except (OSError, RuntimeError):
                candidate = raw_candidate
                project_root = project.resolve()
            if (
                not raw_candidate.is_symlink()
                and candidate.is_relative_to(project_root)
                and candidate.is_file()
                and _sha256_file(candidate) == reward_sha
            ):
                return _load_reward_spec(candidate)
    return _load_final_reward_spec(rewards_dir)


def _select_iter_indices(n_iters: int) -> list[int]:
    """1, N/2, N in 1-based terms — zero-indexed + deduped."""
    if n_iters <= 0:
        return []
    if n_iters == 1:
        return [0]
    if n_iters == 2:
        return [0, 1]
    return sorted({0, n_iters // 2, n_iters - 1})


# ── Label + title PNG rendering (PIL) ───────────────────────────────────
def _find_font(size: int = 18):
    from PIL import ImageFont

    candidates = [
        r"C:/Windows/Fonts/segoeui.ttf",
        r"C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def _render_label_png(text: str, path: Path,
                      width: int = 480, height: int = 56) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color=(10, 10, 14))
    draw = ImageDraw.Draw(img)
    font = _find_font(20)
    # Vertical centering
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_h = 14  # default font fallback
    draw.text((12, max(0, (height - text_h) // 2 - 2)),
              text, font=font, fill=(245, 245, 250))
    img.save(path)
    return path


def _wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        try:
            w = draw.textbbox((0, 0), trial, font=font)[2]
        except AttributeError:
            w = 8 * len(trial)
        if w <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _render_title_png(
    path: Path,
    *,
    behavior_goal: str,
    total_iters: int,
    starting_metric: float | None,
    ending_metric: float | None,
    primary_key: str,
    adapter_class: str,
    width: int,
    height: int = 480,
) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color=(8, 8, 12))
    draw = ImageDraw.Draw(img)

    title_font = _find_font(34)
    h1_font = _find_font(22)
    body_font = _find_font(18)

    pad = 40
    y = pad
    draw.text((pad, y), "Reward Sculptor — final report",
              font=title_font, fill=(240, 240, 250))
    y += 48

    # Behavior goal (wrapped)
    draw.text((pad, y), "behavior goal", font=h1_font,
              fill=(150, 180, 255))
    y += 28
    for line in _wrap_text(draw, f"“{behavior_goal}”", body_font, width - 2 * pad):
        draw.text((pad, y), line, font=body_font, fill=(230, 230, 240))
        y += 24

    y += 16
    start_s = f"{starting_metric:+.3f}" if starting_metric is not None else "n/a"
    end_s = f"{ending_metric:+.3f}" if ending_metric is not None else "n/a"
    for label, value in [
        ("iterations", str(total_iters)),
        (f"{primary_key} (start → end)", f"{start_s} → {end_s}"),
        ("adapter", adapter_class),
    ]:
        draw.text((pad, y), label, font=h1_font, fill=(150, 180, 255))
        draw.text((pad + 360, y), value, font=h1_font, fill=(230, 230, 240))
        y += 30

    img.save(path)
    return path


# ── ffmpeg invocation ───────────────────────────────────────────────────
def _ffmpeg_exe() -> str | None:
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _probe_video_ok(path: Path) -> bool:
    """Cheap validity check: file exists and is > 2KB. Our gym_sb3 adapter
    writes sentinel bytes when ffmpeg isn't available during rollout; those
    sentinels are single-digit KB."""
    return path.is_file() and path.stat().st_size > 2048


def _run_ffmpeg(cmd: list[str]) -> tuple[int, str, str]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def _build_final_mp4(
    *,
    panel_videos: list[Path],
    panel_labels: list[str],
    title_png: Path,
    out_path: Path,
    panel_size: int = 480,
    title_seconds: int = 4,
    fps: int = 25,
) -> tuple[bool, str]:
    """One ffmpeg invocation: each panel video → scaled + overlaid with its
    label PNG → hstack → prepended with a 4s still from `title_png`.

    Returns (ok, stderr). `ok=False` means the final mp4 is missing or
    empty after the call; caller can decide what to do.
    """
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        return False, "ffmpeg not found (system PATH or imageio-ffmpeg)"
    n = len(panel_videos)
    if n == 0:
        return False, "no panel videos"
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-render label PNGs per panel. We keep them in out_path's parent.
    tmp_dir = out_path.parent / ".sculpt_timelapse"
    tmp_dir.mkdir(exist_ok=True)
    label_pngs: list[Path] = []
    for i, text in enumerate(panel_labels):
        p = tmp_dir / f"label_{i}.png"
        _render_label_png(text, p, width=panel_size, height=56)
        label_pngs.append(p)

    hstack_width = panel_size * n

    # Build -i args: [0]=title, [1..]=alternating (label_png, video)
    inputs: list[str] = []
    inputs += ["-loop", "1", "-t", str(title_seconds), "-i", str(title_png)]
    for label_png, video in zip(label_pngs, panel_videos):
        inputs += ["-i", str(label_png), "-i", str(video)]

    # Filtergraph:
    filter_parts: list[str] = []
    # title normalization (resize to hstack width, fix fps)
    filter_parts.append(
        f"[0:v]scale={hstack_width}:{panel_size},"
        f"setsar=1,fps={fps}[title]")

    panel_tags: list[str] = []
    for i in range(n):
        label_idx = 1 + 2 * i
        video_idx = 1 + 2 * i + 1
        v = f"v{i}"
        label_tag = f"l{i}"
        filter_parts.append(
            f"[{video_idx}:v]scale={panel_size}:{panel_size},"
            f"setsar=1,fps={fps}[{v}]")
        filter_parts.append(
            f"[{label_idx}:v]scale={panel_size}:56[{label_tag}]")
        filter_parts.append(f"[{v}][{label_tag}]overlay=0:H-h[p{i}]")
        panel_tags.append(f"[p{i}]")

    if n >= 2:
        filter_parts.append("".join(panel_tags) + f"hstack=inputs={n}[panels]")
    else:
        filter_parts.append(f"{panel_tags[0]}null[panels]")

    filter_parts.append("[title][panels]concat=n=2:v=1:a=0[out]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(out_path),
    ]
    code, _out, err = _run_ffmpeg(cmd)
    ok = (code == 0 and out_path.exists() and out_path.stat().st_size > 0)
    return ok, err


# ── Analytics for the MD report ─────────────────────────────────────────
@dataclass
class _EditSummary:
    iter_index: int
    target_term: str
    operation: str
    rationale: str
    suggested_value: str | None
    paper_refs: list[str]
    requires_env_extension: bool
    delta: float | None  # post-iter metric - pre-iter metric (may be None)


def _collect_iter_edits(
    iter_dirs: list[Path], metric_history: list[float],
) -> list[_EditSummary]:
    """Flatten every diagnosis's proposed_edits into a single list, each
    edit annotated with the primary-metric delta of the iteration AFTER it
    was applied (i.e., metric[i+1] - metric[i]). For the last iter, delta
    is None (not yet measured)."""
    out: list[_EditSummary] = []
    for position, d in enumerate(iter_dirs):
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if not m:
            continue
        i = int(m.group(1))
        diag = _load_json(d / "diagnosis.json")
        if not diag:
            continue
        # delta = metric[i+1] - metric[i] when both exist
        delta = None
        if len(metric_history) > position + 1:
            try:
                delta = (
                    float(metric_history[position + 1])
                    - float(metric_history[position])
                )
            except Exception:  # noqa: BLE001
                delta = None
        for e in diag.get("proposed_edits", []) or []:
            out.append(_EditSummary(
                iter_index=i,
                target_term=e.get("target_term", ""),
                operation=e.get("operation", ""),
                rationale=e.get("rationale", ""),
                suggested_value=e.get("suggested_value"),
                paper_refs=list(e.get("paper_refs", []) or []),
                requires_env_extension=bool(e.get("requires_env_extension", False)),
                delta=delta,
            ))
    return out


def _describe_behavior(behavior: dict, behavior_metric_names: list[str]) -> str:
    """Format behavior.json as a one-paragraph description."""
    if not behavior:
        return "(behavior.json missing)"
    keys = behavior_metric_names or list(behavior.keys())
    parts: list[str] = []
    for k in keys:
        v = behavior.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            parts.append(f"**{k}** = {float(v):.4g}")
        elif isinstance(v, dict):
            # e.g., termination_reason_counts
            inner = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
            parts.append(f"**{k}** = {{{inner}}}")
        else:
            parts.append(f"**{k}** = {v}")
    mr = behavior.get("mean_return")
    if mr is not None and "mean_return" not in (behavior_metric_names or []):
        parts.insert(0, f"**mean_return** = {float(mr):.4g}")
    return "; ".join(parts) if parts else "(empty behavior.json)"


# ── Markdown report ─────────────────────────────────────────────────────
def _write_final_report_md(
    *,
    project: Path,
    config: dict,
    behavior_goal: str,
    iter_dirs: list[Path],
    metric_history: list[float],
    edits: list[_EditSummary],
    final_reward_path: Path,
    final_reward_spec: dict,
    provenance: dict,
    final_mp4_path: Path,
    final_mp4_ok: bool,
    selected_iter_dir: Path | None = None,
) -> Path:
    iter_cfg = config.get("iteration", {}) or {}
    primary_key = str(iter_cfg.get("primary_metric", "mean_return"))
    behavior_metric_names = list(iter_cfg.get("behavior_metrics", []))
    adapter_class = (config.get("adapter", {}) or {}).get("class", "(unknown)")

    first_iter = iter_dirs[0] if iter_dirs else None
    last_iter = selected_iter_dir or (iter_dirs[-1] if iter_dirs else None)
    selection_claimed = selected_iter_dir is not None
    first_behavior = _load_json(first_iter / "rollout" / "behavior.json") if first_iter else {}
    last_behavior = _load_json(last_iter / "rollout" / "behavior.json") if last_iter else {}

    first_index = _iter_number(first_iter) if first_iter is not None else -1
    selected_position = _iter_number(last_iter) if last_iter is not None else -1
    starting_metric = (
        metric_history[first_index]
        if 0 <= first_index < len(metric_history)
        else None
    )
    ending_metric = (
        metric_history[selected_position]
        if 0 <= selected_position < len(metric_history)
        else None
    )

    # Top-3 impactful edits
    ranked = [e for e in edits if e.delta is not None
              and not e.requires_env_extension]
    ranked.sort(key=lambda e: (-(e.delta or 0.0), e.iter_index, e.target_term))
    top3 = ranked[:3]

    # Literature map from provenance (active entries by target term)
    final_hparams = set(
        (final_reward_spec.get("hyperparameters") or {}).keys())
    literature_map: dict[str, list[dict]] = {}
    for term, entries in provenance.items():
        active = [e for e in entries if e.get("still_active")]
        if active:
            literature_map[term] = active

    # Candidate novel contributions: applied edits with empty paper_refs +
    # rationale starting with "novel." (case-insensitive).
    novel: list[_EditSummary] = [
        e for e in edits
        if not e.paper_refs
        and not e.requires_env_extension
        and e.rationale.strip().lower().startswith("novel.")
    ]

    # Summary table rows
    summary_rows: list[dict] = []
    for d in iter_dirs:
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if not m:
            continue
        i = int(m.group(1))
        diag = _load_json(d / "diagnosis.json")
        applied = [e for e in (diag.get("proposed_edits", []) or [])
                   if not e.get("requires_env_extension")]
        n_refs = sum(len(e.get("paper_refs") or []) for e in applied)
        n_novel = sum(1 for e in applied if not (e.get("paper_refs") or []))
        summary_rows.append({
            "iter": i,
            "metric": (
                metric_history[i]
                if i < len(metric_history)
                else None
            ),
            "num_references_added": n_refs,
            "num_novel_edits": n_novel,
        })

    # ── assemble markdown ─────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# Sculpt Final Report\n")
    lines.append(f"- **Behavior goal**: _{behavior_goal}_")
    lines.append(f"- **Adapter**: `{adapter_class}`")
    lines.append(f"- **Project**: `{project}`")
    lines.append(f"- **Iterations completed**: {len(iter_dirs)}")
    if selection_claimed:
        lines.append(
            "- **Selection authority**: canonical receipt + independent "
            "objective proof verified"
        )
    else:
        lines.append(
            "- **Selection authority**: unavailable — this is descriptive run "
            "history and makes no selected-policy or task-success claim"
        )
    lines.append(
        f"- **Primary metric (`{primary_key}`)**: "
        f"{starting_metric:+.4f} → {ending_metric:+.4f} "
        f"(Δ {(ending_metric - starting_metric):+.4f})"
        if starting_metric is not None and ending_metric is not None
        else f"- **Primary metric (`{primary_key}`)**: n/a"
    )
    reward_label = (
        "Selected policy reward module"
        if selection_claimed else "Latest completed-run reward module"
    )
    lines.append(f"- **{reward_label}**: "
                 f"[`rewards/{final_reward_path.name}`](rewards/{final_reward_path.name})  "
                 f"(version `{final_reward_spec.get('version', '?')}`)")
    video_status = "[final.mp4](" + str(final_mp4_path.as_posix()) + ")" if final_mp4_ok else \
        f"_video build failed; see stderr above (expected at_ `{final_mp4_path.as_posix()}`)"
    lines.append(f"- **Time-lapse video**: {video_status}\n")

    # Behavior before/after
    lines.append("## Behavior: starting vs ending\n")
    lines.append(f"**Starting** (iter {_iter_number(first_iter) if first_iter else '?'}): "
                 f"{_describe_behavior(first_behavior, behavior_metric_names)}")
    lines.append("")
    ending_label = "Selected" if selection_claimed else "Latest completed (not selected)"
    lines.append(f"**{ending_label}** (iter {_iter_number(last_iter) if last_iter else '?'}): "
                 f"{_describe_behavior(last_behavior, behavior_metric_names)}\n")

    # Top 3 most impactful edits
    lines.append("## Top 3 most impactful edits (by primary-metric delta)\n")
    if not top3:
        lines.append("_No edits with measurable impact (need at least "
                     "two iterations with a metric change)._\n")
    else:
        lines.append(f"Δ is measured as `{primary_key}[iter+1] - "
                     f"{primary_key}[iter]` — positive means the reward "
                     "change produced improvement on the NEXT iteration.\n")
        for rank, e in enumerate(top3, 1):
            delta_s = f"{e.delta:+.4f}" if e.delta is not None else "n/a"
            refs_s = (", ".join(f"arXiv:{a}" for a in e.paper_refs)
                      if e.paper_refs else "(novel)")
            lines.append(f"{rank}. **iter {e.iter_index} / `{e.target_term}` "
                         f"[{e.operation}]** — Δ = {delta_s}")
            if e.suggested_value:
                lines.append(f"   - suggested_value: `{e.suggested_value}`")
            lines.append(f"   - rationale: {e.rationale}")
            lines.append(f"   - paper_refs: {refs_s}")
        lines.append("")

    # Literature map
    lines.append("## Literature map (final reward)\n")
    if not literature_map:
        lines.append("_No active literature-grounded entries in "
                     "reports/provenance.json._\n")
    else:
        for term, entries in sorted(literature_map.items()):
            marker = "" if term in final_hparams else "  _(target is a "\
                "component/derived term, not a top-level hyperparameter)_"
            lines.append(f"### `{term}`{marker}")
            for entry in entries:
                arxiv_id = entry.get("arxiv_id", "?")
                citation = entry.get("citation") or f"arXiv:{arxiv_id}"
                iter_introduced = entry.get("iter_introduced", "?")
                lines.append(f"- {citation}")
                lines.append(f"  - introduced at iter {iter_introduced}")
                how = (entry.get("how_used") or "").strip()
                if how:
                    lines.append(f"  - how: {how[:400]}"
                                 + ("…" if len(how) > 400 else ""))
            lines.append("")

    # Candidate novel contributions
    lines.append("## Candidate novel contributions\n")
    if not novel:
        lines.append("_No edits flagged as novel in this run._\n")
    else:
        lines.append("These edits were applied without any citation; every "
                     "rationale begins with `novel.` Audit each manually "
                     "before treating as a contribution:\n")
        for e in novel:
            delta_s = f"{e.delta:+.4f}" if e.delta is not None else "n/a"
            lines.append(f"- **iter {e.iter_index} / `{e.target_term}` "
                         f"[{e.operation}]** — Δ {delta_s}")
            if e.suggested_value:
                lines.append(f"  - suggested_value: `{e.suggested_value}`")
            lines.append(f"  - rationale: {e.rationale}")
        lines.append("")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| iter | " + primary_key + " | num_references_added | num_novel_edits |")
    lines.append("|---:|---:|---:|---:|")
    for row in summary_rows:
        metric = row["metric"]
        m_s = f"{metric:+.4f}" if isinstance(metric, (int, float)) else "n/a"
        lines.append(
            f"| {row['iter']} | {m_s} | {row['num_references_added']} | "
            f"{row['num_novel_edits']} |")
    lines.append("")

    # Changelog reference
    cl_path = project / "CHANGELOG.md"
    if cl_path.is_file():
        lines.append("## Changelog\n")
        lines.append(f"Full per-iteration breakdown: [`CHANGELOG.md`]"
                     f"(../{cl_path.name}).")
    lines.append("")

    out = project / "reports" / "final_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _tracked_file(path: Path, root: Path) -> dict[str, Any]:
    """Stable identity for an input that may legitimately be absent."""
    digest = _sha256_file(path)
    try:
        relpath = path.relative_to(root).as_posix()
    except ValueError:
        relpath = path.name
    size = None
    if digest is not None:
        try:
            size = path.stat().st_size
        except OSError:
            digest = None
    return {"path": relpath, "sha256": digest, "bytes": size}


def _iteration_input_snapshot(iter_dir: Path, root: Path) -> dict[str, Any]:
    completion = _completion_receipt(iter_dir)
    if completion is None:  # defensive; callers pre-filter
        raise ValueError(f"iteration is not attested complete: {iter_dir}")
    checkpoint = iter_dir / str(completion["checkpoint"])
    tracked = [
        iter_dir / "iteration_complete.json",
        checkpoint,
        iter_dir / "artifact_tuple.json",
        iter_dir / "fitness.json",
        iter_dir / "metrics.json",
        iter_dir / "reward_spec.json",
        iter_dir / "diagnosis.json",
        iter_dir / "run_context.json",
        Path(str(checkpoint) + ".policy_contract.json"),
        iter_dir / "physical_acceptance_contract.json",
        iter_dir / "physical_acceptance_receipt.json",
        iter_dir / "rollout" / "behavior.json",
        iter_dir / "rollout" / "trajectory.npz",
        iter_dir / "rollout" / "rollout.mp4",
    ]
    return {
        "iter_index": completion["iter_index"],
        "completion": completion,
        "files": [_tracked_file(path, root) for path in tracked],
    }


def _single_project_input_snapshot(project: Path) -> dict[str, Any]:
    completed = _completed_iter_dirs(project / "runs")
    reward_files = sorted(
        (
            path for path in (project / "rewards").glob("v*.py")
            if re.fullmatch(r"v\d+\.py", path.name)
        ),
        key=lambda path: int(path.stem[1:]),
    ) if (project / "rewards").is_dir() else []
    return {
        "config": _tracked_file(project / "config.toml", project),
        "selection": _tracked_file(
            project / "reports" / "selection.json", project,
        ),
        "metric_history": _tracked_file(
            project / "reports" / "metric_history.json", project,
        ),
        "provenance": _tracked_file(
            project / "reports" / "provenance.json", project,
        ),
        "rewards": [_tracked_file(path, project) for path in reward_files],
        "iterations": [
            _iteration_input_snapshot(iter_dir, project)
            for iter_dir in completed
        ],
    }


def report_input_snapshot(
    source_root: Path | str,
    *,
    source_kind: str = "project",
) -> dict[str, Any]:
    """Return the exact report inputs used for staleness detection."""
    root = Path(source_root).resolve()
    if source_kind == "project":
        return {
            "schema": REPORT_RECEIPT_SCHEMA,
            "source_kind": "project",
            "project": _single_project_input_snapshot(root),
        }
    if source_kind != "mission":
        raise ValueError(f"unknown report source kind: {source_kind!r}")
    stages_root = root / "stages"
    stages: list[dict[str, Any]] = []
    if stages_root.is_dir():
        for stage_dir in sorted(stages_root.iterdir(), key=lambda p: p.name):
            if stage_dir.is_dir() and not stage_dir.is_symlink():
                stages.append({
                    "stage": stage_dir.name,
                    "inputs": _single_project_input_snapshot(stage_dir),
                })
    return {
        "schema": REPORT_RECEIPT_SCHEMA,
        "source_kind": "mission",
        "mission": _tracked_file(root / "mission.json", root),
        "stages": stages,
    }


def _write_report_receipt(
    source_root: Path,
    *,
    source_kind: str,
    report_path: Path,
    mp4_path: Path,
    selection_authority: dict[str, Any] | None,
) -> Path:
    snapshot = report_input_snapshot(source_root, source_kind=source_kind)
    selected = (
        selection_authority.get("selected_iter_index")
        if isinstance(selection_authority, dict)
        and selection_authority.get("status") == "verified"
        else None
    )
    receipt = {
        "schema": REPORT_RECEIPT_SCHEMA,
        "source_kind": source_kind,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_sha256": _sha256_file(report_path),
        "final_mp4_sha256": _sha256_file(mp4_path),
        "input_snapshot": snapshot,
        "input_digest": _canonical_digest(snapshot),
        "claim": {
            "status": "verified" if selected is not None else "descriptive_only",
            "selected_iter_index": selected,
            "selection_authority_digest": (
                selection_authority.get("authority_digest")
                if selected is not None else None
            ),
            "selection_receipt_sha256": (
                selection_authority.get("selection_receipt_sha256")
                if selected is not None else None
            ),
            "objective_evidence_sha256": (
                selection_authority.get("objective_evidence_sha256")
                if selected is not None else None
            ),
            "objective_evidence_receipt": (
                selection_authority.get("objective_evidence_receipt")
                if selected is not None else None
            ),
            "claim_inputs": (
                selection_authority.get("claim_inputs")
                if selected is not None else None
            ),
            "claim_inputs_sha256": (
                selection_authority.get("claim_inputs_sha256")
                if selected is not None else None
            ),
        },
    }
    receipt_path = source_root / "reports" / REPORT_RECEIPT_NAME
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return receipt_path


def inspect_report_state(
    source_root: Path | str,
    *,
    source_kind: str = "project",
) -> dict[str, Any]:
    """Classify a retained report as missing/current/stale, fail closed."""
    root = Path(source_root).resolve()
    report_path = root / "reports" / "final_report.md"
    if not report_path.is_file() or report_path.is_symlink():
        return {
            "state": "missing", "reason": "report has not been built",
            "claim_status": "unavailable", "selected_iter_index": None,
        }
    receipt_path = root / "reports" / REPORT_RECEIPT_NAME
    receipt = _load_json(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("schema") != REPORT_RECEIPT_SCHEMA:
        return {
            "state": "stale",
            "reason": "retained report predates input-bound report receipts",
            "claim_status": "unavailable",
            "selected_iter_index": None,
        }
    claim = receipt.get("claim")
    claim = claim if isinstance(claim, dict) else {}
    base = {
        "claim_status": claim.get("status", "unavailable"),
        "selected_iter_index": claim.get("selected_iter_index"),
        "generated_at": receipt.get("generated_at"),
    }
    if (
        receipt.get("source_kind") != source_kind
        or receipt.get("report_sha256") != _sha256_file(report_path)
        or receipt.get("final_mp4_sha256")
        != _sha256_file(root / "reports" / "final.mp4")
    ):
        return {
            "state": "stale",
            "reason": "report bytes or source identity differ from its receipt",
            **base,
        }
    try:
        current = report_input_snapshot(root, source_kind=source_kind)
        current_digest = _canonical_digest(current)
    except Exception as exc:  # noqa: BLE001 - status reads fail closed
        return {
            "state": "stale",
            "reason": f"current report inputs could not be verified: {type(exc).__name__}",
            **base,
        }
    if receipt.get("input_digest") != current_digest:
        return {
            "state": "stale",
            "reason": "selection, evidence, run, reward, or artifact inputs changed",
            **base,
        }
    return {"state": "current", "reason": None, **base}


# ── Public entry ────────────────────────────────────────────────────────
@dataclass
class ReportResult:
    final_mp4_path: Path
    final_mp4_ok: bool
    final_report_md_path: Path
    ffmpeg_stderr: str = ""
    selected_iter_indices: list[int] = dataclasses.field(default_factory=list)
    report_receipt_path: Path | None = None
    report_claim_status: str = "descriptive_only"


def build_report(
    config_path: Path | str,
    out_mp4: Path | str,
    *,
    selection_authority: dict[str, Any] | None = None,
) -> ReportResult:
    """Produce final.mp4 + final_report.md. `config_path` is a Sculptor
    project config.toml; `out_mp4` is where the video is written."""
    config_path = Path(config_path).resolve()
    cfg = _parse_toml(config_path)
    project = config_path.parent
    runs_dir = project / "runs"
    rewards_dir = project / "rewards"
    iter_dirs = _completed_iter_dirs(runs_dir)

    # Ferret out behavior_goal. Prefer the latest iteration's diagnosis.json,
    # fall back to [target].name in config.
    behavior_goal = ""
    for d in reversed(iter_dirs):
        diag = _load_json(d / "diagnosis.json")
        if diag.get("behavior_goal"):
            behavior_goal = str(diag["behavior_goal"])
            break
    if not behavior_goal:
        behavior_goal = (cfg.get("target", {}) or {}).get("name", "(unknown)")

    # Metric history (written by sculpt_run).
    metric_history_obj = _load_json(project / "reports" / "metric_history.json")
    metric_history: list[float] = list(metric_history_obj.get("history", []))

    # A selected-policy claim is earned only by the byte-bound authority the
    # backend issues after canonical selection + objective-proof verification.
    selected_iter_dir = _select_report_iter_dir(
        project, iter_dirs, selection_authority,
    )
    final_reward_path, final_reward_spec = _load_selected_reward_spec(
        project, selected_iter_dir, rewards_dir)

    # Provenance
    provenance = _load_json(project / "reports" / "provenance.json") or {}

    # Select iters for the video + gather label info
    selected_positions = _select_iter_indices(len(iter_dirs))
    selected = [_iter_number(iter_dirs[position]) for position in selected_positions]
    primary_key = str((cfg.get("iteration", {}) or {})
                      .get("primary_metric", "mean_return"))

    panel_videos: list[Path] = []
    panel_labels: list[str] = []
    for position in selected_positions:
        d = iter_dirs[position]
        iter_number = _iter_number(d)
        mp4 = d / "rollout" / "rollout.mp4"
        if not _probe_video_ok(mp4):
            sys.stderr.write(
                f"[report] iter_{iter_number}'s rollout.mp4 missing or too small "
                f"({mp4}); dropping from time-lapse.\n")
            continue
        metric = (
            metric_history[iter_number]
            if 0 <= iter_number < len(metric_history) else None
        )
        if metric is None:
            label = f"Iter {iter_number}"
        else:
            label = f"Iter {iter_number}   {primary_key}={metric:+.3f}"
        panel_videos.append(mp4)
        panel_labels.append(label)

    out_mp4 = Path(out_mp4).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Build the title card size to match panel layout.
    panel_size = 480
    title_width = max(panel_size, panel_size * max(1, len(panel_videos)))
    tmp_dir = out_mp4.parent / ".sculpt_timelapse"
    tmp_dir.mkdir(exist_ok=True)
    title_png = tmp_dir / "title.png"
    _render_title_png(
        title_png,
        behavior_goal=behavior_goal,
        total_iters=len(iter_dirs),
        starting_metric=(
            metric_history[_iter_number(iter_dirs[0])]
            if iter_dirs
            and 0 <= _iter_number(iter_dirs[0]) < len(metric_history)
            else None
        ),
        ending_metric=(
            metric_history[_iter_number(selected_iter_dir or iter_dirs[-1])]
            if iter_dirs
            and 0 <= _iter_number(selected_iter_dir or iter_dirs[-1]) < len(metric_history)
            else None
        ),
        primary_key=primary_key,
        adapter_class=(cfg.get("adapter", {}) or {}).get("class", "(unknown)"),
        width=title_width, height=480,
    )

    # Build the mp4.
    mp4_ok = False
    stderr = ""
    if panel_videos:
        mp4_ok, stderr = _build_final_mp4(
            panel_videos=panel_videos, panel_labels=panel_labels,
            title_png=title_png, out_path=out_mp4, panel_size=panel_size)
    else:
        sys.stderr.write(
            "[report] no valid rollout videos; writing title-card-only mp4.\n")
        # Build a ~4s title-only mp4 so the CLI still produces a file.
        ffmpeg = _ffmpeg_exe()
        if ffmpeg is not None:
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-loop", "1", "-t", "4", "-i", str(title_png),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                str(out_mp4),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            stderr = r.stderr
            mp4_ok = (r.returncode == 0 and out_mp4.exists()
                      and out_mp4.stat().st_size > 0)

    # Always emit the markdown report.
    edits = _collect_iter_edits(iter_dirs, metric_history)
    md_path = _write_final_report_md(
        project=project, config=cfg, behavior_goal=behavior_goal,
        iter_dirs=iter_dirs, metric_history=metric_history,
        edits=edits, final_reward_path=final_reward_path,
        final_reward_spec=final_reward_spec, provenance=provenance,
        final_mp4_path=out_mp4, final_mp4_ok=mp4_ok,
        selected_iter_dir=selected_iter_dir,
    )
    receipt_path = _write_report_receipt(
        project,
        source_kind="project",
        report_path=md_path,
        mp4_path=out_mp4,
        selection_authority=(
            selection_authority if selected_iter_dir is not None else None
        ),
    )

    return ReportResult(
        final_mp4_path=out_mp4, final_mp4_ok=mp4_ok,
        final_report_md_path=md_path, ffmpeg_stderr=stderr,
        selected_iter_indices=selected,
        report_receipt_path=receipt_path,
        report_claim_status=(
            "verified" if selected_iter_dir is not None else "descriptive_only"
        ),
    )


# ── Mission-aware report (§chunk C1) ────────────────────────────────────
# A Mission (sculptor.mission.Mission) decomposes a complex goal into an
# ordered sequence of Stages, each scaffolded as its own mini-project at
# `<mission_dir>/stages/<stage_name>/` — own config.toml, runs/, rewards/,
# reports/. `build_mission_report` walks the stages in order and reuses
# every per-run helper above (unchanged), rather than re-implementing the
# analytics. `build_report` itself is untouched by any of this.
@dataclass
class _StageReportData:
    """Everything one stage contributes to the mission report — the
    stage-scoped analog of the locals `build_report` computes inline."""

    stage_name: str
    stage_dir: Path
    config: dict
    iter_dirs: list[Path]
    metric_history: list[float]
    edits: list[_EditSummary]
    final_reward_path: Path
    final_reward_spec: dict
    provenance: dict
    primary_key: str
    behavior_metric_names: list[str]
    adapter_class: str
    load_error: str = ""  # non-empty when config.toml itself was unreadable


def _collect_stage_data(stage_dir: Path) -> _StageReportData:
    """Gather one stage's report analytics. Mirrors the first half of
    `build_report`'s body, scoped to `stage_dir` instead of a top-level
    project. Never raises — a stage that hasn't scaffolded yet (or whose
    config.toml is unreadable) comes back with empty collections and
    `load_error` set, so the caller can render a graceful placeholder
    section instead of aborting the whole mission report."""
    config_path = stage_dir / "config.toml"
    try:
        cfg = _parse_toml(config_path)
    except Exception as e:  # noqa: BLE001 — stage not scaffolded / corrupt
        return _StageReportData(
            stage_name=stage_dir.name, stage_dir=stage_dir, config={},
            iter_dirs=[], metric_history=[], edits=[],
            final_reward_path=stage_dir / "rewards" / "v0.py",
            final_reward_spec={}, provenance={}, primary_key="mean_return",
            behavior_metric_names=[], adapter_class="(unknown)",
            load_error=f"{type(e).__name__}: {e}",
        )

    runs_dir = stage_dir / "runs"
    rewards_dir = stage_dir / "rewards"
    iter_dirs = _completed_iter_dirs(runs_dir)

    metric_history_obj = _load_json(stage_dir / "reports" / "metric_history.json")
    metric_history: list[float] = list(metric_history_obj.get("history", []))

    final_reward_path, final_reward_spec = _load_final_reward_spec(rewards_dir)
    provenance = _load_json(stage_dir / "reports" / "provenance.json") or {}
    edits = _collect_iter_edits(iter_dirs, metric_history)

    iter_cfg = cfg.get("iteration", {}) or {}
    return _StageReportData(
        stage_name=stage_dir.name, stage_dir=stage_dir, config=cfg,
        iter_dirs=iter_dirs, metric_history=metric_history, edits=edits,
        final_reward_path=final_reward_path,
        final_reward_spec=final_reward_spec, provenance=provenance,
        primary_key=str(iter_cfg.get("primary_metric", "mean_return")),
        behavior_metric_names=list(iter_cfg.get("behavior_metrics", [])),
        adapter_class=(cfg.get("adapter", {}) or {}).get("class", "(unknown)"),
    )


def _stage_status_line(stage) -> str:  # stage: sculptor.mission.Stage
    """One-line status summary for the mission-header stage table."""
    bits = [f"status=`{stage.status}`"]
    if stage.best_metric is not None:
        bits.append(f"best_metric={stage.best_metric:+.4f}")
    bits.append(f"iterations_used={stage.iterations_used}")
    if stage.parent_stage:
        bits.append(f"parent=`{stage.parent_stage}`")
    return ", ".join(bits)


def _write_mission_report_section(
    lines: list[str], *, stage, data: _StageReportData,
) -> None:  # stage: sculptor.mission.Stage
    """Append one stage's section (mirrors `_write_final_report_md`'s
    per-run layout: behavior start→end, top edits, summary table) to
    the running `lines` buffer. Never raises on missing artifacts —
    every read below already tolerates absence via `_load_json`."""
    lines.append(f"## Stage: `{stage.name}`\n")
    lines.append(f"- **Goal**: _{stage.goal_text}_")
    lines.append(f"- **Success criterion**: `{stage.success_criterion}`")
    lines.append(f"- **{_stage_status_line(stage)}**")
    if data.load_error:
        lines.append(
            f"\n_Stage has not scaffolded yet or its config.toml is "
            f"unreadable ({data.load_error}); no run data to report._\n"
        )
        return

    iter_dirs = data.iter_dirs
    metric_history = data.metric_history
    primary_key = data.primary_key
    first_index = _iter_number(iter_dirs[0]) if iter_dirs else -1
    last_index = _iter_number(iter_dirs[-1]) if iter_dirs else -1
    starting_metric = (
        metric_history[first_index]
        if 0 <= first_index < len(metric_history) else None
    )
    ending_metric = (
        metric_history[last_index]
        if 0 <= last_index < len(metric_history) else None
    )

    lines.append(f"- **Adapter**: `{data.adapter_class}`")
    lines.append(f"- **Iterations completed**: {len(iter_dirs)}")
    lines.append(
        f"- **Primary metric (`{primary_key}`)**: "
        f"{starting_metric:+.4f} → {ending_metric:+.4f} "
        f"(Δ {(ending_metric - starting_metric):+.4f})"
        if starting_metric is not None and ending_metric is not None
        else f"- **Primary metric (`{primary_key}`)**: n/a"
    )
    lines.append("")

    if not iter_dirs:
        lines.append("_No completed iterations in this stage yet._\n")
        return

    first_behavior = _load_json(iter_dirs[0] / "rollout" / "behavior.json")
    last_behavior = _load_json(iter_dirs[-1] / "rollout" / "behavior.json")
    lines.append("**Behavior: starting vs ending**\n")
    lines.append(f"- Starting (iter {first_index}): "
                 f"{_describe_behavior(first_behavior, data.behavior_metric_names)}")
    lines.append(f"- Ending (iter {last_index}; not a selected-policy claim): "
                 f"{_describe_behavior(last_behavior, data.behavior_metric_names)}\n")

    ranked = [e for e in data.edits if e.delta is not None
              and not e.requires_env_extension]
    ranked.sort(key=lambda e: (-(e.delta or 0.0), e.iter_index, e.target_term))
    top3 = ranked[:3]
    lines.append("**Top edits (by primary-metric delta)**\n")
    if not top3:
        lines.append("_No edits with measurable impact._\n")
    else:
        for rank, e in enumerate(top3, 1):
            delta_s = f"{e.delta:+.4f}" if e.delta is not None else "n/a"
            refs_s = (", ".join(f"arXiv:{a}" for a in e.paper_refs)
                      if e.paper_refs else "(novel)")
            lines.append(f"{rank}. iter {e.iter_index} / `{e.target_term}` "
                         f"[{e.operation}] — Δ = {delta_s} — {refs_s}")
        lines.append("")

    lines.append("**Summary**\n")
    lines.append("| iter | " + primary_key + " |")
    lines.append("|---:|---:|")
    for d in iter_dirs:
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if not m:
            continue
        i = int(m.group(1))
        metric = metric_history[i] if 0 <= i < len(metric_history) else None
        m_s = f"{metric:+.4f}" if isinstance(metric, (int, float)) else "n/a"
        lines.append(f"| {i} | {m_s} |")
    lines.append("")


def _write_mission_report_md(
    *, mission, mission_dir: Path, stage_data: dict[str, _StageReportData],
    final_mp4_path: Path, final_mp4_ok: bool,
) -> Path:  # mission: sculptor.mission.Mission
    lines: list[str] = []
    lines.append("# Sculpt Mission Report\n")
    lines.append(f"- **Goal**: _{mission.goal}_")
    lines.append(f"- **Mission directory**: `{mission_dir}`")
    lines.append(f"- **Stages**: {len(mission.stages)}")
    lines.append(f"- **Decomposition model**: `{mission.decomposition_model}`")
    video_status = (
        "[final.mp4](" + str(final_mp4_path.as_posix()) + ")" if final_mp4_ok
        else f"_video build failed or skipped; expected at_ "
             f"`{final_mp4_path.as_posix()}`"
    )
    lines.append(f"- **Time-lapse video**: {video_status}\n")

    lines.append("## Decomposition rationale\n")
    lines.append(f"{mission.decomposition_rationale}\n")

    lines.append("## Stages overview\n")
    lines.append("| # | stage | status | best_metric | criterion |")
    lines.append("|---:|---|---|---:|---|")
    for i, stage in enumerate(mission.stages):
        best = f"{stage.best_metric:+.4f}" if stage.best_metric is not None else "n/a"
        crit = stage.success_criterion
        if len(crit) > 60:
            crit = crit[:57] + "…"
        lines.append(f"| {i} | `{stage.name}` | {stage.status} | {best} | `{crit}` |")
    lines.append("")

    for stage in mission.stages:
        data = stage_data[stage.name]
        _write_mission_report_section(lines, stage=stage, data=data)

    out = mission_dir / "reports" / "final_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


_MAX_PANELS_PER_STAGE = 3


def _select_mission_panels(
    stage_data_ordered: list[_StageReportData],
) -> tuple[list[Path], list[str]]:
    """Pick ≤`_MAX_PANELS_PER_STAGE` rollout videos per stage, in stage
    order, using the same `_select_iter_indices` + `_probe_video_ok`
    gating `build_report` uses per-run. Labels: "<stage> · iter N ·
    <primary_key>=<value>" (or without the metric when unavailable)."""
    panel_videos: list[Path] = []
    panel_labels: list[str] = []
    for data in stage_data_ordered:
        if data.load_error or not data.iter_dirs:
            continue
        n = len(data.iter_dirs)
        # Reuse the existing 1/mid/N selector, then cap to the per-stage
        # panel budget (it already returns ≤3 for n<=... but for larger
        # n it also returns exactly 3, so this cap is defensive).
        selected = _select_iter_indices(n)[:_MAX_PANELS_PER_STAGE]
        for position in selected:
            d = data.iter_dirs[position]
            iter_index = _iter_number(d)
            mp4 = d / "rollout" / "rollout.mp4"
            if not _probe_video_ok(mp4):
                sys.stderr.write(
                    f"[mission report] stage {data.stage_name!r} "
                    f"iter_{iter_index}'s "
                    f"rollout.mp4 missing or too small ({mp4}); dropping "
                    f"from time-lapse.\n")
                continue
            metric = (
                data.metric_history[iter_index]
                if 0 <= iter_index < len(data.metric_history) else None
            )
            if metric is None:
                label = f"{data.stage_name} · iter {iter_index}"
            else:
                label = (
                    f"{data.stage_name} · iter {iter_index} · "
                    f"{data.primary_key}={metric:+.3f}"
                )
            panel_videos.append(mp4)
            panel_labels.append(label)
    return panel_videos, panel_labels


def build_mission_report(
    mission_dir: Path | str, out_mp4: Path | str,
) -> ReportResult:
    """Produce `<mission_dir>/reports/final.mp4` + `final_report.md` for
    a multi-stage Mission. `mission_dir` is the directory holding
    `mission.json` (see `sculptor.mission.load_mission` / `save_mission`
    disk layout); `out_mp4` is where the stitched time-lapse is written
    (conventionally `<mission_dir>/reports/final.mp4`, matching the
    project-level `build_report`'s convention).

    Walks stages IN ORDER, reusing the exact per-run helpers `build_report`
    uses (`_find_iter_dirs`, `_collect_iter_edits`, `_select_iter_indices`,
    `_build_final_mp4`, …) pointed at each stage's own mini-project dir.
    A stage that hasn't scaffolded yet, or is missing an optional artifact
    (metric_history.json, provenance.json, a rollout mp4, …), is skipped
    gracefully rather than raising — the report renders what's available."""
    from sculptor.mission import load_mission  # local: avoid import cycle

    mission_dir = Path(mission_dir).resolve()
    mission = load_mission(mission_dir)

    # §mission-persistence increment 1: this loop is status-agnostic by
    # design — it walks EVERY entry in `mission.stages`, including ones
    # marked "superseded" (retained after a redecomposition splice, see
    # `sculpt._maybe_redecompose_and_splice`). A superseded stage has a
    # real on-disk stage_dir with real trained iterations (that's the
    # whole point of retaining it instead of discarding it), so
    # `_collect_stage_data` picks up its footage/metrics the same as
    # any other stage — no special-casing needed, and nothing here
    # raises on a superseded stage's presence.
    stage_data: dict[str, _StageReportData] = {}
    for stage in mission.stages:
        try:
            stage_dir = mission.stage_dir(stage.name)
        except (RuntimeError, KeyError):
            stage_dir = mission_dir / "stages" / stage.name
        stage_data[stage.name] = _collect_stage_data(stage_dir)

    stage_data_ordered = [stage_data[s.name] for s in mission.stages]
    panel_videos, panel_labels = _select_mission_panels(stage_data_ordered)

    out_mp4 = Path(out_mp4).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    panel_size = 480
    title_width = max(panel_size, panel_size * max(1, len(panel_videos)))
    tmp_dir = out_mp4.parent / ".sculpt_timelapse"
    tmp_dir.mkdir(exist_ok=True)
    title_png = tmp_dir / "title.png"

    # Mission-level "primary metric" for the title card: first stage with
    # a non-empty metric history that also has a steering/primary key.
    first_with_history = next(
        (d for d in stage_data_ordered if d.metric_history), None)
    total_iters = sum(len(d.iter_dirs) for d in stage_data_ordered)
    _render_title_png(
        title_png,
        behavior_goal=mission.goal,
        total_iters=total_iters,
        starting_metric=(
            stage_data_ordered[0].metric_history[
                _iter_number(stage_data_ordered[0].iter_dirs[0])
            ]
            if stage_data_ordered and stage_data_ordered[0].iter_dirs
            and _iter_number(stage_data_ordered[0].iter_dirs[0])
            < len(stage_data_ordered[0].metric_history)
            else None
        ),
        ending_metric=(
            stage_data_ordered[-1].metric_history[
                _iter_number(stage_data_ordered[-1].iter_dirs[-1])
            ]
            if stage_data_ordered and stage_data_ordered[-1].iter_dirs
            and _iter_number(stage_data_ordered[-1].iter_dirs[-1])
            < len(stage_data_ordered[-1].metric_history)
            else None
        ),
        primary_key=(first_with_history.primary_key
                    if first_with_history is not None else "mean_return"),
        adapter_class=(stage_data_ordered[0].adapter_class
                       if stage_data_ordered else "(unknown)"),
        width=title_width, height=480,
    )

    mp4_ok = False
    stderr = ""
    if panel_videos:
        mp4_ok, stderr = _build_final_mp4(
            panel_videos=panel_videos, panel_labels=panel_labels,
            title_png=title_png, out_path=out_mp4, panel_size=panel_size)
    else:
        sys.stderr.write(
            "[mission report] no valid rollout videos across any stage; "
            "writing title-card-only mp4.\n")
        ffmpeg = _ffmpeg_exe()
        if ffmpeg is not None:
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-loop", "1", "-t", "4", "-i", str(title_png),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                str(out_mp4),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            stderr = r.stderr
            mp4_ok = (r.returncode == 0 and out_mp4.exists()
                      and out_mp4.stat().st_size > 0)

    md_path = _write_mission_report_md(
        mission=mission, mission_dir=mission_dir, stage_data=stage_data,
        final_mp4_path=out_mp4, final_mp4_ok=mp4_ok,
    )
    receipt_path = _write_report_receipt(
        mission_dir,
        source_kind="mission",
        report_path=md_path,
        mp4_path=out_mp4,
        selection_authority=None,
    )

    # selected_iter_indices has no single-run meaning for a mission; report
    # the per-stage-ordered index list flattened (0-based within each
    # stage) so callers inspecting it still get something meaningful.
    selected_indices: list[int] = []
    for data in stage_data_ordered:
        if data.load_error or not data.iter_dirs:
            continue
        selected_indices.extend(
            _iter_number(data.iter_dirs[position])
            for position in _select_iter_indices(
                len(data.iter_dirs),
            )[:_MAX_PANELS_PER_STAGE]
        )

    return ReportResult(
        final_mp4_path=out_mp4, final_mp4_ok=mp4_ok,
        final_report_md_path=md_path, ffmpeg_stderr=stderr,
        selected_iter_indices=selected_indices,
        report_receipt_path=receipt_path,
        report_claim_status="descriptive_only",
    )
