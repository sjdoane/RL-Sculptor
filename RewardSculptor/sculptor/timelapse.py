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
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def _load_final_reward_spec(rewards_dir: Path) -> tuple[Path, dict]:
    """Return (path_to_latest_vN.py, REWARD_SPEC)."""
    best: tuple[int, Path] | None = None
    for p in rewards_dir.glob("v*.py"):
        m = re.fullmatch(r"v(\d+)", p.stem)
        if m:
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, p)
    if best is None:
        return rewards_dir / "v0.py", {}
    path = best[1]
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


def _find_iter_dirs(runs_dir: Path) -> list[Path]:
    dirs: list[tuple[int, Path]] = []
    if not runs_dir.is_dir():
        return []
    for d in runs_dir.iterdir():
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if m and d.is_dir():
            dirs.append((int(m.group(1)), d))
    dirs.sort(key=lambda x: x[0])
    return [d for _, d in dirs]


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
        l = f"l{i}"
        filter_parts.append(
            f"[{video_idx}:v]scale={panel_size}:{panel_size},"
            f"setsar=1,fps={fps}[{v}]")
        filter_parts.append(f"[{label_idx}:v]scale={panel_size}:56[{l}]")
        filter_parts.append(f"[{v}][{l}]overlay=0:H-h[p{i}]")
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
    for d in iter_dirs:
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if not m:
            continue
        i = int(m.group(1))
        diag = _load_json(d / "diagnosis.json")
        if not diag:
            continue
        # delta = metric[i+1] - metric[i] when both exist
        delta = None
        if len(metric_history) > i + 1 and len(metric_history) > i:
            try:
                delta = float(metric_history[i + 1]) - float(metric_history[i])
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
) -> Path:
    iter_cfg = config.get("iteration", {}) or {}
    primary_key = str(iter_cfg.get("primary_metric", "mean_return"))
    behavior_metric_names = list(iter_cfg.get("behavior_metrics", []))
    adapter_class = (config.get("adapter", {}) or {}).get("class", "(unknown)")

    first_iter = iter_dirs[0] if iter_dirs else None
    last_iter = iter_dirs[-1] if iter_dirs else None
    first_behavior = _load_json(first_iter / "rollout" / "behavior.json") if first_iter else {}
    last_behavior = _load_json(last_iter / "rollout" / "behavior.json") if last_iter else {}

    starting_metric = metric_history[0] if metric_history else None
    ending_metric = metric_history[-1] if metric_history else None

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
            "metric": metric_history[i] if i < len(metric_history) else None,
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
    lines.append(
        f"- **Primary metric (`{primary_key}`)**: "
        f"{starting_metric:+.4f} → {ending_metric:+.4f} "
        f"(Δ {(ending_metric - starting_metric):+.4f})"
        if starting_metric is not None and ending_metric is not None
        else f"- **Primary metric (`{primary_key}`)**: n/a"
    )
    lines.append(f"- **Final reward module**: "
                 f"[`rewards/{final_reward_path.name}`](rewards/{final_reward_path.name})  "
                 f"(version `{final_reward_spec.get('version', '?')}`)")
    video_status = "[final.mp4](" + str(final_mp4_path.as_posix()) + ")" if final_mp4_ok else \
        f"_video build failed; see stderr above (expected at_ `{final_mp4_path.as_posix()}`)"
    lines.append(f"- **Time-lapse video**: {video_status}\n")

    # Behavior before/after
    lines.append("## Behavior: starting vs ending\n")
    lines.append(f"**Starting** (iter {0}): "
                 f"{_describe_behavior(first_behavior, behavior_metric_names)}")
    lines.append("")
    lines.append(f"**Ending** (iter {len(iter_dirs) - 1}): "
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


# ── Public entry ────────────────────────────────────────────────────────
@dataclass
class ReportResult:
    final_mp4_path: Path
    final_mp4_ok: bool
    final_report_md_path: Path
    ffmpeg_stderr: str = ""
    selected_iter_indices: list[int] = dataclasses.field(default_factory=list)


def build_report(
    config_path: Path | str, out_mp4: Path | str,
) -> ReportResult:
    """Produce final.mp4 + final_report.md. `config_path` is a Sculptor
    project config.toml; `out_mp4` is where the video is written."""
    config_path = Path(config_path).resolve()
    cfg = _parse_toml(config_path)
    project = config_path.parent
    runs_dir = project / "runs"
    rewards_dir = project / "rewards"
    iter_dirs = _find_iter_dirs(runs_dir)

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

    # Final reward
    final_reward_path, final_reward_spec = _load_final_reward_spec(rewards_dir)

    # Provenance
    provenance = _load_json(project / "reports" / "provenance.json") or {}

    # Select iters for the video + gather label info
    selected = _select_iter_indices(len(iter_dirs))
    primary_key = str((cfg.get("iteration", {}) or {})
                      .get("primary_metric", "mean_return"))

    panel_videos: list[Path] = []
    panel_labels: list[str] = []
    for idx in selected:
        d = iter_dirs[idx]
        mp4 = d / "rollout" / "rollout.mp4"
        if not _probe_video_ok(mp4):
            sys.stderr.write(
                f"[report] iter_{idx}'s rollout.mp4 missing or too small "
                f"({mp4}); dropping from time-lapse.\n")
            continue
        metric = metric_history[idx] if idx < len(metric_history) else None
        if metric is None:
            label = f"Iter {idx}"
        else:
            label = f"Iter {idx}   {primary_key}={metric:+.3f}"
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
        starting_metric=metric_history[0] if metric_history else None,
        ending_metric=metric_history[-1] if metric_history else None,
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
    )

    return ReportResult(
        final_mp4_path=out_mp4, final_mp4_ok=mp4_ok,
        final_report_md_path=md_path, ffmpeg_stderr=stderr,
        selected_iter_indices=selected,
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
    iter_dirs = _find_iter_dirs(runs_dir)

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
    starting_metric = metric_history[0] if metric_history else None
    ending_metric = metric_history[-1] if metric_history else None

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
    lines.append(f"- Starting (iter 0): "
                 f"{_describe_behavior(first_behavior, data.behavior_metric_names)}")
    lines.append(f"- Ending (iter {len(iter_dirs) - 1}): "
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
        metric = metric_history[i] if i < len(metric_history) else None
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
        for idx in selected:
            d = data.iter_dirs[idx]
            mp4 = d / "rollout" / "rollout.mp4"
            if not _probe_video_ok(mp4):
                sys.stderr.write(
                    f"[mission report] stage {data.stage_name!r} iter_{idx}'s "
                    f"rollout.mp4 missing or too small ({mp4}); dropping "
                    f"from time-lapse.\n")
                continue
            metric = (data.metric_history[idx]
                      if idx < len(data.metric_history) else None)
            if metric is None:
                label = f"{data.stage_name} · iter {idx}"
            else:
                label = f"{data.stage_name} · iter {idx} · {data.primary_key}={metric:+.3f}"
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
        starting_metric=(stage_data_ordered[0].metric_history[0]
                         if stage_data_ordered and stage_data_ordered[0].metric_history
                         else None),
        ending_metric=(stage_data_ordered[-1].metric_history[-1]
                       if stage_data_ordered and stage_data_ordered[-1].metric_history
                       else None),
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

    # selected_iter_indices has no single-run meaning for a mission; report
    # the per-stage-ordered index list flattened (0-based within each
    # stage) so callers inspecting it still get something meaningful.
    selected_indices: list[int] = []
    for data in stage_data_ordered:
        if data.load_error or not data.iter_dirs:
            continue
        selected_indices.extend(
            _select_iter_indices(len(data.iter_dirs))[:_MAX_PANELS_PER_STAGE])

    return ReportResult(
        final_mp4_path=out_mp4, final_mp4_ok=mp4_ok,
        final_report_md_path=md_path, ffmpeg_stderr=stderr,
        selected_iter_indices=selected_indices,
    )
