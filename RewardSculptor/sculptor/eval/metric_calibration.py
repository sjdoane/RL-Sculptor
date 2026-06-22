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
import re
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


#: JSON-mode authoring budget. The author emits a full CompetenceLadder /
#: GamingArchetypeSet JSON (nested rungs/groups) + adaptive thinking — generous so a
#: complete object fits (a truncated one fails to parse → recorded skip, never silent).
_AUTHOR_MAX_TOKENS = 12000


def _extract_json_obj(text: str) -> dict:
    """Extract the first top-level JSON object from a model completion — tolerant of a
    ```json fence or surrounding prose. Raises ValueError if none parses (the caller
    records a skip; a malformed/truncated author NEVER grants)."""
    if not text or not text.strip():
        raise ValueError("empty author response")
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = m.group(1) if m else text
    start = candidate.find("{")
    if start < 0:
        raise ValueError("no JSON object in author response")
    obj, _ = json.JSONDecoder().raw_decode(candidate[start:])
    if not isinstance(obj, dict):
        raise ValueError("author response is not a JSON object")
    return obj


def _author_structured(client: Any, model: str, system_prompt: str,
                       payload: dict, schema: Any) -> Any:
    """Author a structured pydantic object via a PLAIN completion + JSON parse — NOT
    `messages.parse` / constrained decoding.

    WHY: the CompetenceLadder / GamingArchetypeSet schemas are too deeply nested (a
    list of Groups, each with a RoleQuery + a `float | list[float]` Union scalar) for
    the API's grammar compiler — `messages.parse(output_format=...)` 400s 'schema is
    too complex' / hangs 'grammar compilation timed out', so the constrained-decode
    path made task-derived calibration (the novel-task STEERING payoff) NEVER grant.
    JSON-mode sidesteps the grammar compiler entirely and preserves the full
    vocabulary. The schema is pinned IN-PROMPT (model_json_schema) so key names/types
    are exact. Raises on any failure (the caller records a skip — never grants)."""
    instr = (
        system_prompt
        + "\n\nOUTPUT FORMAT: respond with ONLY a single JSON object (no prose, no "
          "markdown fence) that conforms to this JSON Schema:\n"
        + json.dumps(schema.model_json_schema(), separators=(",", ":")))
    resp = client.messages.create(
        model=model,
        max_tokens=_AUTHOR_MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=instr,
        messages=[{"role": "user",
                   "content": json.dumps(payload, indent=2, default=str)}],
    )
    if getattr(resp, "stop_reason", None) == "max_tokens":
        raise ValueError(
            f"author response truncated at max_tokens={_AUTHOR_MAX_TOKENS}")
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return schema.model_validate(_extract_json_obj(text))


def _author_ladder(client: Any, model: str, payload: dict) -> Any:
    """One blind ladder-author call → a CompetenceLadder. The author is given
    ONLY the goal/robot/joint_names/style/vocabulary — never any metric."""
    from sculptor.eval.ladder_synth import CompetenceLadder
    from sculptor.prompts import load_prompt

    return _author_structured(client, model, load_prompt("gen_competence_ladder"),
                              payload, CompetenceLadder)


# ── §Ship 53: adversarial gaming archetypes (L3) ─────────────────────────
#: An INDEPENDENT, metric-blind author proposes ~N OFF-GOAL "gaming policies"
#: (degenerate behaviors a naive metric scores high). Each is rendered with the
#: Ship-51 synthesizer and scored; the metric is GAMEABLE iff any gaming policy
#: scores in COMPETENT territory. Generalizes Ship-47's hard-coded walker/flail
#: negatives to ANY task. Validated against the built-ins (which score plausible
#: gaming ≤0.15 vs competent 0.37-0.76, ratio ≤0.33) and a gameable raw-|jv|
#: metric (tremor-gaming 1.0 vs competent 0.03): a 0.6×competent relative ceiling
#: with a 0.5 absolute backstop separates them with headroom.
_ADV_REL_CEIL = 0.6      # gaming must score below 0.6 × the competent reference
_ADV_ABS_CEIL = 0.5      # AND below 0.5 absolute (no gaming may look half-competent)
_ADV_N = 3               # archetypes the blind author proposes


def _author_gaming(client: Any, model: str, payload: dict) -> Any:
    """One blind gaming-archetype author call → a GamingArchetypeSet. Given ONLY
    the goal/robot/joint_names — never any metric (same firewall as the ladder
    author). JSON-mode (see `_author_structured`) — the GamingArchetypeSet schema is
    too complex for the API grammar compiler."""
    from sculptor.eval.ladder_synth import GamingArchetypeSet
    from sculptor.prompts import load_prompt

    return _author_structured(client, model, load_prompt("gen_gaming_archetypes"),
                              payload, GamingArchetypeSet)


# ── §Metric-quality laws (LAW 9 / completeness D5): deterministic kick-family ──
#: required-losers. The documented g1-kick-v5 hacks as DETERMINISTIC negative
#: probes, rendered WITH left/right_foot_pos_b — the signed forward-direction
#: channel the LLM/`render_rung` archetype path CANNOT exercise (render_rung emits
#: no foot data, so the kick-behind hack is invisible to it; that gap is exactly
#: why spec_g1_kick was never adversarially caught on direction). Together the
#: probes give the per-channel coverage obligation (≥1 probe per scored channel)
#: by construction: direction (kick-behind), completion (one-leg / whip-and-fall),
#: amplitude (partial rep). Name-parameterized — they resolve the LEFT sagittal
#: leg BY NAME (one source of truth: leg_sagittal_indices' LEG_SAGITTAL_* + a
#: left-side filter) so a single builder serves the 12-joint synthetic body AND a
#: real 29-DOF G1. Sibling of metric_validate._archetypes' kick hacks (locked to
#: the 12-joint battery); kept here so the calibration gate is robot-agnostic.
_KL_T, _KL_E, _KL_DT = 120, 4, 0.02
#: Tokens that make a kick's goal-axis or support-mode AMBIGUOUS. A frame-scoped
#: loser (direction / support) is DROPPED — not injected — when present, so a
#: novel kick variant (mule / spin / roundhouse / one-leg) is never false-denied
#: (LAW 0). The gate's ABSOLUTE ceiling denies HARDER than metric_validate's
#: RELATIVE bar (hv ≥ kick_pos), so conservative high-confidence framing matters.
_KICK_LATERAL_TOKENS = ("spin", "spinning", "roundhouse", "side", "sideways",
                        "lateral", "sweep", "crescent", "hook", "circle", "round")
_KICK_SINGLE_TOKENS = ("one", "single", "flamingo", "stork", "balancing", "balance")


def _kick_upright_g() -> np.ndarray:
    g = np.zeros((_KL_T, _KL_E, 3)); g[..., 2] = -1.0
    return g


def _kick_standing_root() -> np.ndarray:
    r = np.zeros((_KL_T, _KL_E, 3)); r[..., 2] = 0.7
    return r


def _kick_foot_swing(direction: float) -> tuple[np.ndarray, np.ndarray]:
    """Left-foot anterior (pelvis-frame x) swing — forward (+1) for a real kick,
    rearward (−1) for the kick-behind hack; the right (stance) foot stays put.
    §kick-fix: 0.40 m peak (was 0.30) so the competent reference clears the new
    spec_g1_kick forward-foot excursion gate (~1.0) and competent_ref stays ~0.78;
    the partial-rep loser (this × 0.2 = 0.08 m) stays well below the gate floor."""
    lf = np.zeros((_KL_T, _KL_E, 3)); rf = np.zeros((_KL_T, _KL_E, 3))
    for start in range(20, _KL_T, 40):
        for k in range(10):
            if start + k < _KL_T:
                lf[start + k, :, 0] = direction * 0.40 * np.sin(np.pi * k / 10)
    return lf, rf


def _kick_pack(jp, jv, g, root, lf=None, rf=None) -> dict:
    d = {"joint_pos": jp, "joint_vel": jv,
         "projected_gravity_b": g, "root_link_pos_w": root}
    if lf is not None:
        d["left_foot_pos_b"] = lf
    if rf is not None:
        d["right_foot_pos_b"] = rf
    return d


def _kick_behavior() -> dict:
    return {"max_episode_steps": _KL_T, "rollout_num_envs": _KL_E,
            "step_dt": _KL_DT}


def _left_sagittal_leg(joint_names: list[str]) -> list[int]:
    """The LEFT sagittal-plane leg joints (hip pitch + knee + ankle pitch),
    excluding hip roll/yaw — the swing leg of a forward kick. One source of
    truth with the metrics (LEG_SAGITTAL_* + a left filter)."""
    from sculptor.eval.joint_resolver import (
        LEG_SAGITTAL_AXES,
        LEG_SAGITTAL_SEGMENTS,
        select_joints,
    )

    return select_joints(list(joint_names), segments=LEG_SAGITTAL_SEGMENTS,
                         axes=LEG_SAGITTAL_AXES, sides=["left"])


def _kick_competent_reference(
    joint_names: list[str],
) -> Optional[tuple[dict, dict, dict]]:
    """A deterministic, genuinely-competent forward kick (peak-8.0 sagittal-leg
    bursts, forward foot swing, upright, stationary) — the most-charitable
    competence anchor the adversarial ceiling is measured against (it scores
    spec_g1_kick ≈ 0.78). Returns (arrays, behavior, meta), or None when the left
    sagittal leg can't be resolved on this robot (no anchor → skip the kick gate,
    never false-deny)."""
    legs = _left_sagittal_leg(joint_names)
    if not legs:
        return None
    J = len(joint_names)
    jv = np.zeros((_KL_T, _KL_E, J))
    for start in range(20, _KL_T, 40):
        for j in legs:
            jv[start:start + 5, :, j] = 8.0
    jp = np.cumsum(jv, axis=0) * _KL_DT
    lf, rf = _kick_foot_swing(+1.0)
    arrays = _kick_pack(jp, jv, _kick_upright_g(), _kick_standing_root(), lf, rf)
    return arrays, _kick_behavior(), {"joint_names": list(joint_names)}


def kick_required_losers(
    joint_names: list[str], behavior_goal: str,
    robot_hint: Optional[str] = None,
) -> list[dict]:
    """Deterministic kick gaming probes WITH foot_pos_b — the kick hack rungs in
    the adversarial negative set. Each entry is
    `{name, channel, arrays, behavior, meta}`. Frame-scoped CONSERVATIVELY
    (LAW 0): a directional / support loser is DROPPED when the goal frame is
    ambiguous, so a novel kick variant is never false-denied. Returns [] when the
    left sagittal leg can't be resolved (unknown robot → coverage gap recorded by
    the gate, never a wrong-joint deny). Never raises."""
    legs = _left_sagittal_leg(joint_names)
    if not legs:
        return []
    from sculptor.eval.metric_validate import resolve_goal_frame

    frame = resolve_goal_frame(behavior_goal, robot_hint)
    tokens = set(re.findall(r"[a-z]+", (behavior_goal or "").lower()))
    forward_ok = (frame.get("goal_axis") == "+x"
                  and not (tokens & set(_KICK_LATERAL_TOKENS)))
    double_ok = (frame.get("support_mode") == "double"
                 and not (tokens & set(_KICK_SINGLE_TOKENS)))

    J = len(joint_names)
    beh = _kick_behavior()
    meta = {"joint_names": list(joint_names)}
    rng = np.random.default_rng(0)
    losers: list[dict] = []

    # partial_kick (AMPLITUDE) — a single brief, sub-floor flick with a tiny
    # forward foot: correctly directed but neither completed nor full-amplitude.
    # ALWAYS (a kick is, frame-independently, a completed full-amplitude motion).
    jvp = np.zeros((_KL_T, _KL_E, J))
    for j in legs:
        jvp[20:23, :, j] = 3.0
    jpp = np.cumsum(jvp, axis=0) * _KL_DT
    lfp, rfp = _kick_foot_swing(+1.0)
    losers.append({"name": "partial_kick", "channel": "amplitude",
                   "arrays": _kick_pack(jpp, jvp, _kick_upright_g(),
                                        _kick_standing_root(), lfp * 0.2, rfp),
                   "behavior": beh, "meta": dict(meta)})

    # whip_and_fall (COMPLETION) — a violent burst whose uprightness collapses;
    # the completion gate's upright term must floor it. ALWAYS (a kick never
    # topples — true for every kick frame).
    jvw = np.zeros((_KL_T, _KL_E, J))
    for start in range(20, _KL_T, 40):
        for j in legs:
            jvw[start:start + 5, :, j] = 12.0
    jpw = np.cumsum(jvw, axis=0) * _KL_DT
    gw = np.zeros((_KL_T, _KL_E, 3))
    cut = _KL_T // 4
    gw[:cut, :, 2] = -1.0
    gw[cut:, :, 0] = 1.0
    lfw, rfw = _kick_foot_swing(+1.0)
    losers.append({"name": "whip_and_fall", "channel": "completion",
                   "arrays": _kick_pack(jpw, jvw, gw, _kick_standing_root(),
                                        lfw, rfw),
                   "behavior": beh, "meta": dict(meta)})

    # active_kick_behind (DIRECTION) — full bursts, REARWARD foot. Only on an
    # UNAMBIGUOUSLY forward goal (the rear-direction gate must not fire on a mule
    # / spin / roundhouse kick).
    if forward_ok:
        jvk = np.zeros((_KL_T, _KL_E, J))
        for start in range(20, _KL_T, 40):
            for j in legs:
                jvk[start:start + 5, :, j] = 8.0
        jpk = np.cumsum(jvk, axis=0) * _KL_DT
        lfb, rfb = _kick_foot_swing(-1.0)
        losers.append({"name": "active_kick_behind", "channel": "direction",
                       "arrays": _kick_pack(jpk, jvk, _kick_upright_g(),
                                            _kick_standing_root(), lfb, rfb),
                       "behavior": beh, "meta": dict(meta)})

    # one_leg_balance (COMPLETION/support) — sub-threshold wiggle, foot held
    # forward static, no launch. Only on an UNAMBIGUOUSLY double-support goal.
    if double_ok:
        jvol = rng.normal(0, 0.3, (_KL_T, _KL_E, J))
        jpol = np.cumsum(jvol, axis=0) * _KL_DT
        lf_hold = np.zeros((_KL_T, _KL_E, 3))
        lf_hold[..., 0] = 0.20
        losers.append({"name": "one_leg_balance", "channel": "completion",
                       "arrays": _kick_pack(jpol, jvol, _kick_upright_g(),
                                            _kick_standing_root(), lf_hold,
                                            np.zeros((_KL_T, _KL_E, 3))),
                       "behavior": beh, "meta": dict(meta)})

    return losers


#: The scored channels a kick metric must defend (the per-channel coverage
#: obligation). intensity is excluded — a weak-but-forward kick is partial
#: competence, NOT a gaming policy.
_KICK_SCORED_CHANNELS = ("direction", "completion", "amplitude")


def adversarial_archetype_gate(
    gen_fn: Any,
    roles: Any,
    joint_names: list[str],
    competent_ref: float,
    *,
    client: Any,
    model: str = _TD_MODEL_ID,
    base_payload: Optional[dict] = None,
    metric_src: str = "",
    n_archetypes: int = _ADV_N,
    rel_ceil: float = _ADV_REL_CEIL,
    abs_ceil: float = _ADV_ABS_CEIL,
    required_losers: Optional[list[dict]] = None,
    scored_channels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """§Ship 53 (L3): an INDEPENDENT, metric-blind author proposes `n_archetypes`
    OFF-GOAL gaming policies for the goal; each is rendered (Ship-51 synthesizer)
    and scored by the metric. The metric is GAMEABLE iff some gaming policy scores
    in competent territory — at or above `min(rel_ceil·competent_ref, abs_ceil)`.

    §Metric-quality laws (LAW 9): `required_losers` are DETERMINISTIC gaming probes
    (the documented kick hacks, WITH foot_pos_b — the direction channel `render_rung`
    can't render) scored IN THEIR OWN guarded loop that runs REGARDLESS of the LLM
    author outcome — so the metric is always exercised against the curated hacks
    even when the author call fails/echoes/yields nothing (the v5-anchor fix). A
    required-loser at/above the ceiling makes the metric gameable, exactly like an
    LLM archetype. `scored_channels` records the per-channel coverage obligation
    (≥1 probe per channel); a GAP is a FLAG (`coverage_gaps`), NEVER a deny. With
    both args unset the function is BYTE-IDENTICAL to the Ship-53 behavior.

    NEVER raises and NEVER denies on ABSENCE of evidence (an author crash, a
    leaked/echoed payload, or zero renderable archetypes is inconclusive, not a
    deny); a denial requires POSITIVE evidence of a gaming policy beating
    competence. Provenance (payload/context/response hashes, per-probe scores) is
    recorded for meta.json. Mirrors the ladder author's anti-collusion disciplines:
    the metric source is NEVER in the payload (hard self-check) and a set that
    echoes the metric source is dropped (soft guard); the min-Spearman / source-leak
    anti-collusion guards are untouched."""
    from sculptor.eval.ladder_synth import render_rung

    base_payload = base_payload or {}
    payload = {**base_payload, "n_archetypes": n_archetypes}
    rec: dict[str, Any] = {
        "ran": False, "ok": True, "gameable": False, "reason": None,
        "competent_ref": round(float(competent_ref), 4),
        "model_id": model, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rel_ceil": rel_ceil, "abs_ceil": abs_ceil,
        "payload_sha256": _sha(payload), "context_sha256": _sha(base_payload),
        "archetypes": [], "required_losers": [],
    }

    worst, worst_name = 0.0, None

    # ── 1. DETERMINISTIC required-losers — scored FIRST and INDEPENDENTLY of the
    #     LLM author, so the metric is probed against the curated kick hacks even
    #     if the author call fails / echoes / yields nothing. The role injection is
    #     CRITICAL: a role-based metric (reads meta['joint_roles']) scores 0 on a
    #     loser whose roles weren't resolved, masking a real gameability.
    losers_scored = 0
    for loser in (required_losers or []):
        name = loser.get("name", "loser")
        try:
            meta = dict(loser.get("meta") or {})
            inject_joint_roles(meta, roles)
            s = float(gen_fn(loser["arrays"], loser["behavior"], meta)
                      .get("spec_score", 0.0))
        except Exception as e:  # noqa: BLE001 — unscorable probe = no evidence
            rec["required_losers"].append(
                {"name": name, "channel": loser.get("channel"),
                 "skipped": f"score error: {type(e).__name__}"})
            continue
        losers_scored += 1
        rec["required_losers"].append(
            {"name": name, "channel": loser.get("channel"), "score": round(s, 4)})
        if s > worst:
            worst, worst_name = s, name

    # ── 2. LLM gaming-archetype author (best-effort breadth; metric-blind). A
    #     failure / echo / leak is NO evidence — it never denies; the deterministic
    #     losers above still stand.
    author_reason: Optional[str] = None
    gset = None
    payload_text = json.dumps(payload, default=str)
    if "def compute_spec" in payload_text or (
            metric_src and metric_src[:120] in payload_text):
        # HARD anti-collusion self-check: WE built `payload`, so the metric must
        # not appear in it. A hit is a programmer error — skip the author (a bug
        # must never deny), but the deterministic losers still run.
        author_reason = ("adversarial: metric leaked into author payload (bug) — "
                         "author skipped")
    else:
        try:
            gset = _author_gaming(client, model, payload)
        except Exception as e:  # noqa: BLE001 — a failed author call = no evidence
            author_reason = (f"adversarial: author call failed "
                             f"({type(e).__name__}) — inconclusive, not enforced")
    if gset is not None:
        rec["response_sha256"] = _sha(getattr(gset, "model_dump", lambda: {})())
        rec["goal_restated"] = getattr(gset, "goal_restated", "")
        # SOFT anti-collusion: a set that echoes the metric source is dropped.
        gset_text = json.dumps(getattr(gset, "model_dump", lambda: {})(), default=str)
        if metric_src and any(
                metric_src[i:i + 40] in gset_text
                for i in range(0, max(0, len(metric_src) - 40), 40)):
            author_reason = ("adversarial: archetypes echo metric source — "
                             "dropped, not enforced")
            gset = None

    arche_scored = 0
    if gset is not None:
        for i, arch in enumerate(list(getattr(gset, "archetypes", []) or [])[:6]):
            motion = getattr(arch, "motion", None)
            name = getattr(arch, "name", "gaming")
            if motion is None or getattr(motion, "degenerate_axis", False):
                rec["archetypes"].append({"name": name, "skipped": "degenerate_axis"})
                continue
            # Render + role-inject + score under ONE guard so a single malformed
            # archetype degrades to "skipped" (no evidence) rather than propagating.
            try:
                arrays, behavior, meta = render_rung(motion, joint_names, rung_index=500 + i)
                inject_joint_roles(meta, roles)
                s = float(gen_fn(arrays, behavior, meta).get("spec_score", 0.0))
                resolved = meta.get("groups_resolved_counts", {})
            except Exception as e:  # noqa: BLE001 — unrenderable/unscorable = no evidence
                rec["archetypes"].append(
                    {"name": name, "skipped": f"render/score error: {type(e).__name__}"})
                continue
            arche_scored += 1
            rec["archetypes"].append({
                "name": name, "strategy": str(getattr(arch, "strategy", ""))[:160],
                "score": round(s, 4), "resolved": resolved,
            })
            if s > worst:
                worst, worst_name = s, name

    # ── 3. per-channel coverage obligation (LAW 9): record which scored channels a
    #     probe covered. A GAP is a FLAG (never-silent), NEVER a deny — absence of a
    #     probe is absence of evidence. The deterministic kick losers MEET the
    #     obligation by construction (no gaps for a forward double-support kick).
    if scored_channels is not None:
        covered = {l["channel"] for l in rec["required_losers"]
                   if "score" in l and l.get("channel")}
        rec["coverage"] = {c: (c in covered) for c in scored_channels}
        rec["coverage_gaps"] = [c for c in scored_channels if c not in covered]

    # ── 4. verdict.
    rec["worst_gaming"] = round(worst, 4)
    rec["worst_name"] = worst_name
    rec["ran"] = (arche_scored > 0) or (losers_scored > 0)
    if not rec["ran"]:
        rec["reason"] = author_reason or (
            "adversarial: no renderable gaming archetype — inconclusive, not enforced")
        return rec

    ceiling = min(rel_ceil * float(competent_ref), abs_ceil)
    rec["ceiling"] = round(ceiling, 4)
    gameable = (worst >= rel_ceil * float(competent_ref)) or (worst >= abs_ceil)
    rec["gameable"] = bool(gameable)
    rec["ok"] = not gameable
    if gameable:
        rec["reason"] = (
            f"adversarial: gaming policy {worst_name!r} scored {worst:.3f} "
            f"≥ ceiling {ceiling:.3f} (competent {competent_ref:.3f}) — "
            f"metric is gameable")
    elif author_reason:
        # The deterministic losers ran clean but the LLM breadth pass was
        # inconclusive — surface it (never-silent) without changing the verdict.
        rec["author_note"] = author_reason
    return rec


def adversarial_archetype_gate_spec(
    builtin_name: str,
    behavior_goal: str,
    *,
    client: Any = None,
    robot_hint: Optional[str] = None,
    model: str = _TD_MODEL_ID,
    n_archetypes: int = _ADV_N,
    rel_ceil: float = _ADV_REL_CEIL,
    abs_ceil: float = _ADV_ABS_CEIL,
) -> dict[str, Any]:
    """§Metric-quality laws (LAW 9): run the adversarial gaming-archetype gate on a
    HAND-AUTHORED `spec_*` metric — the surface that NEVER existed before, which is
    why the gate never ran on `spec_g1_kick`, the metric that scored g1-kick-v5.

    Resolves the spec fn from `_SPEC_FNS`, builds a deterministic competent
    reference + the kick required-losers (WITH foot_pos_b) for the kick family, and
    scores them. spec fns read `meta['joint_names']` directly, so `roles=[]` (no
    role injection needed); `metric_src=""` (the spec is not LLM-authored, so the
    source-leak guard is moot). NEVER raises; never denies on absence of evidence.

    This is AUDIT-grade for built-ins: the verdict is recorded/surfaced — it must
    NEVER auto-revoke the ground-truth fence (built-ins are the trusted calibration
    anchor). Raises KeyError on an unknown builtin (like `calibrate_metric`)."""
    from sculptor.eval.metric_validate import resolve_behavior_family
    from sculptor.eval.robot_manifest import robot_joint_names as _manifest

    if builtin_name not in _SPEC_FNS:
        raise KeyError(f"unknown built-in metric {builtin_name!r}")
    spec_fn = _SPEC_FNS[builtin_name]
    names = list(_manifest(robot_hint) or _NAMES_12)
    family = resolve_behavior_family(behavior_goal, robot_hint)

    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=2, timeout=240.0)

    losers: list[dict] = []
    channels: Optional[list[str]] = None
    competent_ref = 0.0
    if family == "kick":
        losers = kick_required_losers(names, behavior_goal, robot_hint)
        channels = list(_KICK_SCORED_CHANNELS)
        comp = _kick_competent_reference(names)
        if comp is not None:
            try:
                competent_ref = float(spec_fn(*comp).get("spec_score", 0.0))
            except Exception:  # noqa: BLE001 — no reference → ceiling falls to abs_ceil
                competent_ref = 0.0

    base_payload = {"behavior_goal": behavior_goal, "robot_hint": robot_hint,
                    "joint_names": names}
    rec = adversarial_archetype_gate(
        spec_fn, [], names, competent_ref, client=client, model=model,
        base_payload=base_payload, metric_src="", n_archetypes=n_archetypes,
        rel_ceil=rel_ceil, abs_ceil=abs_ceil,
        required_losers=losers or None, scored_channels=channels)
    rec["builtin"] = builtin_name
    rec["behavior_goal"] = behavior_goal
    rec["family"] = family
    rec["audit_only"] = True   # NEVER revoke the ground-truth fence on a deny.
    return rec


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
    adversarial: bool = False,
    adversarial_n: int = _ADV_N,
    adversarial_required_losers: bool = False,
) -> dict[str, Any]:
    """§Ship 51: earn steer-rights on a NOVEL task (no built-in ground truth)
    by ranking K INDEPENDENTLY-authored competence ladders. Each of K sources
    is a fresh, metric-BLIND LLM call authoring a `CompetenceLadder`; a
    deterministic synthesizer renders it; the metric is scored per rung and
    Spearman-correlated (midrank) against the rung index. Earns the grant iff
    `rho_min ≥ rho_floor` AND `agreement_fraction ≥ agree_floor` AND
    `n_valid ≥ 2` AND the ladders are non-degenerate. NEVER raises — every
    failure mode is a specific observe-only `reason` (the run stays alive).

    §Ship 53 (L3): when `adversarial=True` (flag-gated; default off so the grant
    is unchanged) and the ladders already grant, an INDEPENDENT metric-blind
    author proposes gaming policies (`adversarial_archetype_gate`); a metric that
    scores any off-goal gaming policy in competent territory is GAMEABLE and
    DENIED (an extra task-specific required-loser). The verdict is recorded under
    `adversarial` regardless; absence of evidence never denies.

    Record shape is a SUPERSET of `calibrate_metric` (so metric_store / the
    firewall / the UI need no change): `spearman` mirrors `rho_min`."""
    from sculptor.eval.ladder_synth import render_ladder
    from sculptor.eval.robot_manifest import robot_joint_names as _manifest

    def _record(ok, rho_min, agreement, sources, *, degenerate=False,
                reason=None, n_valid=0, error=None, adversarial=None) -> dict[str, Any]:
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
            "adversarial": adversarial, "sources": sources,
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
    base_ok = (rho_min >= rho_floor) and (agreement >= agree_floor) and (n_valid >= _TD_MIN_VALID)

    # §Ship 53 (L3): only when base_ok do we spend a call probing gameability —
    # the metric's competent reference is its best top-rung score across valid
    # sources (most charitable, hardest to false-deny). A gameable verdict is a
    # task-specific extra required-loser, ANDed into the grant.
    adv = None
    if adversarial and base_ok:
        try:
            competent_ref = max(
                (s["gen_scores"][-1] for s in sources
                 if s.get("ladder_ok") and s.get("gen_scores")), default=0.0)
            # §Metric-quality laws (LAW 9): OPT-IN deterministic kick required-
            # losers (the kick hack rungs WITH foot_pos_b). Default OFF so the grant
            # stays byte-identical to Ship 53; ON only for a novel KICK-VARIANT task
            # (a canonical kick routes to the built-in path, not here). Each loser
            # gets the metric's roles injected inside the gate.
            req_losers = None
            sc = None
            if adversarial_required_losers:
                from sculptor.eval.metric_validate import resolve_behavior_family
                if resolve_behavior_family(behavior_goal, robot_hint) == "kick":
                    req_losers = kick_required_losers(names, behavior_goal, robot_hint)
                    sc = list(_KICK_SCORED_CHANNELS)
            adv = adversarial_archetype_gate(
                gen_fn, roles, names, competent_ref, client=client, model=model,
                base_payload=base_payload, metric_src=metric_src,
                n_archetypes=adversarial_n,
                required_losers=req_losers, scored_channels=sc)
        except Exception:  # noqa: BLE001 — an unexpected gate crash is NO evidence,
            adv = None      # never a deny (calibrate_task_derived never raises)

    adv_denies = bool(adv and adv.get("gameable"))
    ok = base_ok and not adv_denies
    if ok:
        reason = None
    elif adv_denies:
        reason = adv.get("reason")
    elif rho_min < rho_floor:
        reason = (f"task-derived: ladders disagree (rho_min={rho_min:.2f} "
                  f"< {rho_floor:.2f}) — observe-only")
    else:
        reason = (f"task-derived: only {n_agree}/{k_sources} ladders agree "
                  f"(need {agree_floor:.2f}) — observe-only")
    return _record(ok, rho_min, agreement, sources, n_valid=n_valid,
                   reason=reason, adversarial=adv)


# ── §Ship 52: one standardized trust score across both calibration paths ──

_TRUST_W_CAL = 0.6
_TRUST_W_EVID = 0.4


def compute_trust(
    calibration: Optional[dict], validation: Optional[dict] = None,
) -> dict[str, Any]:
    """A single standardized CONFIDENCE scalar (built-in OR task-derived) plus
    the per-layer breakdown the UI shows:

        trust = w_cal·CAL + w_evi·EVID
        CAL   = clip((rho_min − 0.5) / 0.5, 0, 1)        # rank evidence
        EVID  = gate_pass(validate) · gate_pass(axioms) · agreement_fraction

    `rho_min` is the built-in `spearman` for the 5 families (a single ground-
    truth source, agreement 1.0) or the task-derived `rho_min` over K sources.

    DESIGN NOTE: trust is a DISPLAY confidence, NOT the steer gate. The design
    doc's literal "trust ≥ 0.7 gates steering" is internally inconsistent — it
    must BOTH reduce to the built-in rho ≥ 0.7 AND admit the task-derived
    rho_min ≥ 0.5 floor, and no single threshold on this scalar does both (a
    floor task-derived grant has trust ≈ 0.27). So the GRANT stays each path's
    own gate (see `grant_decision`); trust ranks how confident a grant is."""
    cal = calibration or {}
    rho_min = cal.get("rho_min")
    if rho_min is None:
        rho_min = cal.get("spearman", 0.0)
    rho_min = float(rho_min or 0.0)
    agreement = float(cal.get("agreement_fraction", 1.0) or 0.0)
    cal_term = float(np.clip((rho_min - 0.5) / 0.5, 0.0, 1.0))
    v = validation or {}
    gate_validate = bool(v.get("ok", True))
    gate_axioms = bool((v.get("axioms") or {}).get("ok", True))
    evid = (1.0 if gate_validate else 0.0) * (1.0 if gate_axioms else 0.0) * agreement
    trust = _TRUST_W_CAL * cal_term + _TRUST_W_EVID * evid
    return {
        "trust": round(float(trust), 4),
        "cal": round(cal_term, 4), "evid": round(float(evid), 4),
        "rho_min": round(rho_min, 4), "agreement_fraction": round(agreement, 4),
        "method": cal.get("method", "builtin"),
        "gate_validate": gate_validate, "gate_axioms": gate_axioms,
        "w_cal": _TRUST_W_CAL, "w_evid": _TRUST_W_EVID,
    }


def grant_decision(
    calibration: Optional[dict], validation: Optional[dict] = None,
) -> bool:
    """The unified steer-rights grant: the path's own calibration `ok` ANDed
    with validate ∧ axioms. For an ACCEPTED metric validate ∧ axioms are
    already true, so this is BYTE-IDENTICAL to today for the 5 built-ins
    (grant ⟺ rho ≥ 0.7) — the re-assertion is defense-in-depth (a metric whose
    persisted record shows a failed gate can never silently steer)."""
    cal = calibration or {}
    v = validation or {}
    gate_validate = bool(v.get("ok", True))
    gate_axioms = bool((v.get("axioms") or {}).get("ok", True))
    return bool(cal.get("ok")) and gate_validate and gate_axioms
