"""Calibration: does a generated metric RANK policies like the
hand-authored ground-truth metric? (§Ship 35 — the circularity firewall.)

A generated metric is an LLM-authored proxy. Before it may STEER a run
(drive selection/early-stop) it must agree with a hand-authored ground
truth on tasks that ground truth covers: Spearman rank-correlation ≥ a
threshold over a competence ladder. Until then it runs OBSERVE-ONLY. The
4 hand-authored metrics in spec_metrics.py never retire — they are the
permanent calibration fence + regression set.

This offline check uses SYNTHETIC competence ladders (graded rollouts the
ground-truth metric is known to order correctly). Real-policy calibration
(Spearman over a pool of trained policies) is the stronger check and runs
when a GPU is available; this is the no-GPU proxy.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sculptor.eval.generated_metric import (
    inject_joint_roles,
    load_generated_metric,
    read_required_roles,
)
from sculptor.eval.spec_metrics import _SPEC_FNS

T, E, J = 120, 4, 12
_NAMES_12 = [
    "left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_shoulder_pitch", "right_shoulder_pitch",
    "left_elbow", "right_elbow", "torso", "neck",
]
_BEHAVIOR = {"max_episode_steps": T, "rollout_num_envs": E, "step_dt": 0.02}


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rho via Pearson on ranks (numpy only). 0.0 if EITHER side
    has no variation — a constant metric carries no rank information and
    must NOT spuriously correlate (argsort-of-constant yields sequential
    ranks, so guard on the RAW std, not the ranks)."""
    av, bv = np.asarray(a, float), np.asarray(b, float)
    # §Ship 41 review: spec_score is contractually [0,1]; round to 6 decimals
    # BEFORE the variation guard so sub-resolution drift cannot manufacture a
    # monotone rank. A degenerate metric reading joint_pos magnitude scored
    # ~1e-7 across the (Ship-41-enriched, cumsum-joint_pos) ladder; its 1e-7
    # std cleared the 1e-12 guard and argsort then gave a spurious rho=1.0.
    av, bv = np.round(av, 6), np.round(bv, 6)
    # §Ship 35 review: epsilon guard (exact == 0 can miss tiny-but-nonzero
    # std from float noise, spuriously correlating a near-constant metric).
    if av.size < 2 or av.std() < 1e-12 or bv.std() < 1e-12:
        return 0.0
    ra = np.argsort(np.argsort(av)).astype(float)
    rb = np.argsort(np.argsort(bv)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _upright_g(t: int = T) -> np.ndarray:
    g = np.zeros((t, E, 3)); g[..., 2] = -1.0
    return g


def _ladder(builtin_name: str) -> list[tuple[dict, dict, dict]]:
    """Graded competence ladder (low→high) for a built-in metric: rollouts
    the ground-truth metric should score in increasing order."""
    t = np.arange(T)
    meta = {"joint_names": _NAMES_12}

    def loco(speed: float):
        root = np.zeros((T, E, 3)); root[..., 2] = 0.30
        root[..., 0] = (t * speed)[:, None]
        return ({"root_link_pos_w": root, "projected_gravity_b": _upright_g()},
                _BEHAVIOR, meta)

    if builtin_name in ("go1_trot",):
        return [loco(s) for s in (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)]

    if builtin_name == "cartpole_balance":
        out = []
        for ln in (50, 150, 250, 350, 450, 500):
            out.append(({}, {**_BEHAVIOR, "mean_episode_length": ln,
                             "max_episode_steps": 500}, meta))
        return out

    dt = _BEHAVIOR["step_dt"]

    if builtin_name == "g1_kick":
        # §Ship 41: populate ALL physical arrays (stationary, upright, standing
        # height) — the spec needs only joint_vel+gravity, but a generated
        # metric that gates on a stationary base / standing height returns 0.0
        # when root_link_pos_w/joint_pos are absent, so it could NEVER calibrate
        # (Spearman 0). The added arrays don't change the spec's rank order.
        out = []
        root = np.zeros((T, E, 3)); root[..., 2] = 0.7   # stationary, standing
        for strength in (0.0, 1.0, 2.0, 4.0, 8.0):
            jv = np.zeros((T, E, J))
            for start in range(20, T, 40):       # discrete leg bursts
                for jdx in (0, 2, 4):            # left hip/knee/ankle
                    jv[start:start + 5, :, jdx] = strength
            jp = np.cumsum(jv, axis=0) * dt      # consistent integrated position
            out.append(({"joint_vel": jv, "joint_pos": jp,
                         "projected_gravity_b": _upright_g(),
                         "root_link_pos_w": root},
                        _BEHAVIOR, meta))
        return out

    if builtin_name == "g1_floss":
        out = []
        root = np.zeros((T, E, 3)); root[..., 2] = 0.7
        for amp in (0.0, 0.1, 0.2, 0.4):
            jp = np.zeros((T, E, J))
            hip = amp * np.sin(2 * np.pi * t / 25)
            arm = amp * np.sin(2 * np.pi * t / 25 + np.pi)   # anti-phase
            for jdx in (0, 1):
                jp[:, :, jdx] = hip[:, None]
            for jdx in (6, 7, 8, 9):
                jp[:, :, jdx] = arm[:, None]
            jv = np.gradient(jp, axis=0)
            out.append(({"joint_pos": jp, "joint_vel": jv,
                         "projected_gravity_b": _upright_g(),
                         "root_link_pos_w": root},
                        _BEHAVIOR, meta))
        return out

    if builtin_name == "g1_jump":
        # §Ship 41: graded vertical hops (crouch→launch→apex→land) with knee
        # extension bursts, upright, no horizontal travel.
        out = []
        for height in (0.0, 0.1, 0.2, 0.35, 0.5):
            z = np.full(T, 0.55)
            jv = np.zeros((T, E, J))
            for start in range(15, T, 35):
                for k in range(20):
                    if start + k < T:
                        z[start + k] = 0.55 + height * np.sin(np.pi * k / 20)
                        if k < 6:                # launch: knees extend
                            for jdx in (2, 3):
                                jv[start + k, :, jdx] = height * 12.0
            jp = np.cumsum(jv, axis=0) * dt
            root = np.zeros((T, E, 3)); root[..., 2] = z[:, None]
            out.append(({"root_link_pos_w": root, "joint_vel": jv,
                         "joint_pos": jp, "projected_gravity_b": _upright_g()},
                        _BEHAVIOR, meta))
        return out

    raise KeyError(f"no calibration ladder for built-in {builtin_name!r}")


def calibrate_metric(
    generated_module_path: Path | str,
    builtin_name: str,
    *,
    threshold: float = 0.7,
) -> dict[str, Any]:
    """Compute Spearman rank-correlation between a generated metric and a
    built-in ground-truth metric over the latter's competence ladder.
    `ok=True` (steer-rights earned) iff rho ≥ threshold. Never raises."""
    if builtin_name not in _SPEC_FNS:
        raise KeyError(f"unknown built-in metric {builtin_name!r}")
    try:
        gen_fn = load_generated_metric(generated_module_path)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "spearman": 0.0, "threshold": threshold,
                "builtin": builtin_name, "error": f"{type(e).__name__}: {e}"}
    # §Ship 49: resolve the metric's declared joint roles against the
    # synthetic biped names the ladder carries, so a role-based metric reads
    # the right columns (lenient — the 12-joint body has no roll/yaw axes).
    roles = read_required_roles(generated_module_path)
    builtin_fn = _SPEC_FNS[builtin_name]
    ladder = _ladder(builtin_name)
    gen_scores, builtin_scores = [], []
    for arrays, behavior, meta in ladder:
        inject_joint_roles(meta, roles, lenient=True)
        try:
            gen_scores.append(float(gen_fn(arrays, behavior, meta).get("spec_score", 0.0)))
        except Exception:  # noqa: BLE001 — a crash on a ladder point = 0
            gen_scores.append(0.0)
        builtin_scores.append(float(builtin_fn(arrays, behavior, meta).get("spec_score", 0.0)))
    rho = spearman(gen_scores, builtin_scores)
    return {
        "ok": bool(rho >= threshold),
        "spearman": round(rho, 4),
        "threshold": threshold,
        "builtin": builtin_name,
        "n": len(ladder),
        "gen_scores": [round(s, 4) for s in gen_scores],
        "builtin_scores": [round(s, 4) for s in builtin_scores],
    }


# ── §Ship 51: L2 task-derived calibration (the novel-task unblocker) ──────

#: Gate constants (auditable, byte-stable; kwargs-overridable in tests). The
#: per-source bar is 0.8 (NOT 0.5): a SATURATING metric scores midrank ≈0.707
#: vs the rung axis, which 0.5 would wrongly pass — 0.8 cleanly separates a
#: whole-ladder ranking (~0.97-1.0) from single-rung discrimination (0.707).
_TD_PER_SOURCE_THRESH = 0.8
_TD_RHO_FLOOR = 0.5
_TD_AGREE_FLOOR = 2.0 / 3.0
_TD_K_SOURCES = 3
_TD_MIN_VALID = 2
_TD_SPREAD_MIN = 0.15            # author-fault: metric near-constant on the ladder
_TD_SEPARATION_MIN = 0.2         # metric-fault: doesn't beat the degenerate anchor
_TD_MODEL_ID = "claude-opus-4-7"

#: Distinct authoring STYLES across the K sources — reduces correlated phrasing
#: so the K ladders are genuinely independent (anti-collusion).
_TD_STYLES = (
    "Describe each rung by concrete physical magnitudes — joint angles in "
    "radians, base speed, hop height, uprightness fraction.",
    "Describe each rung as an observable behavior a person watching would see, "
    "from total failure up to fluent mastery.",
    "Define each rung by the specific failure of the rung BELOW it that this "
    "rung fixes — the incremental competence gained at each step.",
)


def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with TIES AVERAGED (0-based). numpy-only (no scipy)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_midrank(a: list[float], b: list[float]) -> float:
    """Spearman rho with TIE-AVERAGED ranks — the task-derived path only.

    The argsort `spearman()` gives ties SEQUENTIAL ranks, which spuriously
    scores a saturating metric `[0.1,0.9,0.9,0.9,0.9]` or a last-rung-only
    metric `[.5,.5,.5,.5,.9]` as rho=1.0 against the tie-free rung axis — a
    FALSE GRANT. Midrank scores both 0.707 (below the 0.8 bar) while a
    genuinely monotone metric stays ~0.97-1.0. Keeps the round-6 + std-floor
    anti-spurious guards that killed the historical 1e-7 joint_pos-magnitude
    spurious correlation."""
    av, bv = np.round(np.asarray(a, float), 6), np.round(np.asarray(b, float), 6)
    if av.size < 2 or av.std() < 1e-12 or bv.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(_midrank(av), _midrank(bv))[0, 1])


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _author_ladder(client: Any, model: str, payload: dict) -> Any:
    """One blind ladder-author call → a CompetenceLadder. The author is given
    ONLY the goal/robot/joint_names/style/vocabulary — never any metric."""
    from sculptor.eval.ladder_synth import CompetenceLadder
    from sculptor.prompts import load_prompt

    system_prompt = load_prompt("gen_competence_ladder")
    resp = client.messages.parse(
        model=model,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user",
                   "content": json.dumps(payload, indent=2, default=str)}],
        output_format=CompetenceLadder,
    )
    return resp.parsed_output


def calibrate_task_derived(
    generated_module_path: Path | str,
    behavior_goal: str,
    robot_hint: Optional[str] = None,
    *,
    client: Any = None,
    model: str = _TD_MODEL_ID,
    k_sources: int = _TD_K_SOURCES,
    robot_joint_names: Optional[list[str]] = None,
    per_source_thresh: float = _TD_PER_SOURCE_THRESH,
    rho_floor: float = _TD_RHO_FLOOR,
    agree_floor: float = _TD_AGREE_FLOOR,
) -> dict[str, Any]:
    """§Ship 51: earn steer-rights on a NOVEL task (no built-in ground truth)
    by ranking K INDEPENDENTLY-authored competence ladders. Each of K sources
    is a fresh, metric-BLIND LLM call authoring a `CompetenceLadder`; a
    deterministic synthesizer renders it; the metric is scored per rung and
    Spearman-correlated (midrank) against the rung index. Earns the grant iff
    `rho_min ≥ rho_floor` AND `agreement_fraction ≥ agree_floor` AND
    `n_valid ≥ 2` AND the ladders are non-degenerate. NEVER raises — every
    failure mode is a specific observe-only `reason` (the run stays alive).

    Record shape is a SUPERSET of `calibrate_metric` (so metric_store / the
    firewall / the UI need no change): `spearman` mirrors `rho_min`."""
    from sculptor.eval.ladder_synth import render_ladder
    from sculptor.eval.robot_manifest import robot_joint_names as _manifest

    def _record(ok, rho_min, agreement, sources, *, degenerate=False,
                reason=None, n_valid=0, error=None) -> dict[str, Any]:
        return {
            "ok": bool(ok), "method": "task_derived",
            "spearman": round(float(rho_min), 4),    # mirrors rho_min for the UI
            "rho_min": round(float(rho_min), 4),
            "agreement_fraction": round(float(agreement), 4),
            "k_sources": k_sources, "n_valid": n_valid,
            "builtin": None, "threshold": rho_floor,
            "per_source_thresh": per_source_thresh, "agree_floor": agree_floor,
            "behavior_goal": behavior_goal, "robot_hint": robot_hint,
            "degenerate": bool(degenerate), "reason": reason, "error": error,
            "sources": sources,
        }

    # Load the metric (a load failure is a hard, specific deny).
    try:
        gen_fn = load_generated_metric(generated_module_path)
        roles = read_required_roles(generated_module_path)
    except Exception as e:  # noqa: BLE001
        return _record(False, 0.0, 0.0, [],
                       reason=f"metric failed to load: {type(e).__name__}: {e}",
                       error=f"{type(e).__name__}: {e}")

    names = list(robot_joint_names or _manifest(robot_hint) or [])
    if not names:
        return _record(False, 0.0, 0.0, [],
                       reason="task-derived: unknown robot — no joint manifest "
                              "to render ladders against; observe-only")

    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=2, timeout=240.0)

    metric_src = ""
    try:
        metric_src = Path(generated_module_path).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    # The shared context every source sees — identical across sources (asserted
    # below). The metric source is NEVER part of it.
    base_payload = {"behavior_goal": behavior_goal, "robot_hint": robot_hint,
                    "joint_names": names}
    context_hash = _sha(base_payload)

    sources: list[dict[str, Any]] = []
    valid_rhos: list[float] = []
    n_agree = 0
    for si in range(k_sources):
        style = _TD_STYLES[si % len(_TD_STYLES)]
        payload = {**base_payload, "authoring_style": style,
                   "n_rungs": 4, "source_index": si}
        rec: dict[str, Any] = {
            "style_id": si, "model_id": model,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "payload_sha256": _sha(payload), "context_sha256": _sha(base_payload),
        }
        # Anti-collusion HARD self-check: we built `payload`, so the metric must
        # not appear in it. A failure is a programmer error in THIS function.
        payload_text = json.dumps(payload, default=str)
        if "def compute_spec" in payload_text or (
                metric_src and metric_src[:120] in payload_text):
            rec["skip_reason"] = "metric source leaked into author payload (bug)"
            sources.append(rec)
            continue
        try:
            ladder = _author_ladder(client, model, payload)
        except Exception as e:  # noqa: BLE001 — a failed author call = no evidence
            rec["skip_reason"] = f"author call failed: {type(e).__name__}: {e}"
            sources.append(rec)
            continue
        rec["competence_axis"] = getattr(ladder, "competence_axis", "")
        rec["response_sha256"] = _sha(getattr(ladder, "model_dump", lambda: {})())
        # Anti-collusion SOFT guard: a ladder that echoes the metric source is
        # dropped (a leak attempt), never scored.
        ladder_text = json.dumps(
            getattr(ladder, "model_dump", lambda: {})(), default=str)
        if metric_src and any(
                metric_src[i:i + 40] in ladder_text
                for i in range(0, max(0, len(metric_src) - 40), 40)):
            rec["skip_reason"] = "ladder echoes metric source"
            sources.append(rec)
            continue

        rungs = list(getattr(ladder, "rungs", []) or [])
        rec["n_rungs"] = len(rungs)
        synth = render_ladder(rungs, names)
        if synth["degenerate"]:
            rec["skip_reason"] = synth["reason"]
            rec["degenerate"] = True
            sources.append(rec)
            continue

        gen_scores = []
        for arrays, behavior, meta in synth["rungs"]:
            inject_joint_roles(meta, roles)
            try:
                gen_scores.append(
                    float(gen_fn(arrays, behavior, meta).get("spec_score", 0.0)))
            except Exception:  # noqa: BLE001 — crash on a rung = 0 (penalize)
                gen_scores.append(0.0)
        rec["gen_scores"] = [round(s, 4) for s in gen_scores]

        spread = max(gen_scores) - min(gen_scores)
        distinct = len({round(s, 6) for s in gen_scores})
        # SPREAD/DISTINCT sanity indicts the AUTHOR (no-evidence, not disagreement).
        if spread < _TD_SPREAD_MIN or distinct < 3:
            rec["skip_reason"] = (f"metric near-constant on this ladder "
                                  f"(spread {spread:.3f} < {_TD_SPREAD_MIN})")
            sources.append(rec)
            continue

        rho = spearman_midrank(gen_scores, list(range(len(gen_scores))))
        separation = gen_scores[-1] - gen_scores[0]   # top vs degenerate anchor
        rec["rho"] = round(rho, 4)
        rec["separation"] = round(separation, 4)
        rec["ladder_ok"] = True
        # ABSOLUTE-SEPARATION anchor indicts the METRIC (counts as disagreement).
        if separation < _TD_SEPARATION_MIN:
            rec["skip_reason"] = (f"metric does not separate competent from the "
                                  f"degenerate anchor by ≥{_TD_SEPARATION_MIN} "
                                  f"(got {separation:.3f})")
            valid_rhos.append(rho)        # still a valid source, just failing
            sources.append(rec)
            continue
        valid_rhos.append(rho)
        if rho >= per_source_thresh:
            n_agree += 1
        sources.append(rec)

    n_valid = len(valid_rhos)
    if n_valid < _TD_MIN_VALID:
        responded = sum(1 for s in sources if "rho" in s or "gen_scores" in s)
        return _record(False, 0.0, 0.0, sources, n_valid=n_valid,
                       reason=f"task-derived: only {n_valid} usable ladder(s) "
                              f"(of {k_sources}) — observe-only")
    rho_min = min(valid_rhos)
    agreement = n_agree / float(k_sources)
    ok = (rho_min >= rho_floor) and (agreement >= agree_floor) and (n_valid >= _TD_MIN_VALID)
    if ok:
        reason = None
    elif rho_min < rho_floor:
        reason = (f"task-derived: ladders disagree (rho_min={rho_min:.2f} "
                  f"< {rho_floor:.2f}) — observe-only")
    else:
        reason = (f"task-derived: only {n_agree}/{k_sources} ladders agree "
                  f"(need {agree_floor:.2f}) — observe-only")
    return _record(ok, rho_min, agreement, sources, n_valid=n_valid, reason=reason)
