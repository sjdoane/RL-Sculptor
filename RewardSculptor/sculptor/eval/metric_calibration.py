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
    GENERATED_FN_NAME,
    inject_joint_roles,
    load_generated_metric,
    load_generated_module,
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
    # §round-6 FALSE-GRANT guard: a NaN/inf carries no rank information and must NOT
    # spuriously correlate (argsort sorts NaN to the end → a fake monotone rank, and the
    # std guard below is `nan < 1e-12` == False → bypassed). Reject non-finite outright.
    if not (np.isfinite(av).all() and np.isfinite(bv).all()):
        return 0.0
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
    # §round-11: load ONCE (a single gated exec) and derive both the compute fn
    # AND the roles from the loaded MODULE — read_required_roles(mod) does NOT
    # re-load. This keeps the whole load inside the try (the round-10 loader can
    # now raise on a violation/unreadable file) so the "Never raises" contract
    # holds, and removes a TOCTOU re-read+re-screen window between the two loads.
    try:
        gen_module = load_generated_module(generated_module_path)
        gen_fn = getattr(gen_module, GENERATED_FN_NAME, None)
        if not callable(gen_fn):
            raise ValueError(f"metric lacks a callable {GENERATED_FN_NAME}()")
        # §Ship 49: resolve the metric's declared joint roles against the
        # synthetic biped names the ladder carries, so a role-based metric reads
        # the right columns (lenient — the 12-joint body has no roll/yaw axes).
        roles = read_required_roles(gen_module)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "spearman": 0.0, "threshold": threshold,
                "builtin": builtin_name, "error": f"{type(e).__name__}: {e}"}
    builtin_fn = _SPEC_FNS[builtin_name]
    ladder = _ladder(builtin_name)
    gen_scores, builtin_scores = [], []
    for arrays, behavior, meta in ladder:
        inject_joint_roles(meta, roles, lenient=True)
        try:
            s = float(gen_fn(arrays, behavior, meta).get("spec_score", 0.0))
        except Exception:  # noqa: BLE001 — a crash on a ladder point = 0
            s = 0.0
        gen_scores.append(s if np.isfinite(s) else 0.0)   # §round-6: NaN → 0 (no spurious rho)
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
    # §round-6 FALSE-GRANT guard: non-finite carries no rank info and would slip the
    # std-floor (`nan < 1e-12` == False) and rank as max — reject outright.
    if not (np.isfinite(av).all() and np.isfinite(bv).all()):
        return 0.0
    if av.size < 2 or av.std() < 1e-12 or bv.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(_midrank(av), _midrank(bv))[0, 1])


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


_ECHO_WINDOW = 40
_ECHO_MIN = 12


def _echoes_source(metric_src: str, text: str) -> bool:
    """SOFT anti-collusion tripwire: does `text` (an author's output) verbatim-echo a
    non-trivial chunk of the metric source? Length-ROBUST (round-4 review): a short
    source (≤ window) is tested whole; a long one is scanned by `_ECHO_WINDOW`-char
    windows at HALF-window stride PLUS an explicit tail window, so no region of the
    source is left unscanned (the prior fixed-full-stride scan missed sub-window and
    tail regions). A redundant tripwire — the hard self-check + the metric-blind author
    + the deterministic render/anchor/agreement gates are the load-bearing defenses."""
    if not metric_src or not text:
        return False
    s = metric_src
    if len(s) <= _ECHO_WINDOW:
        return len(s) >= _ECHO_MIN and s.strip() in text
    starts = list(range(0, len(s) - _ECHO_WINDOW, _ECHO_WINDOW // 2))
    starts.append(len(s) - _ECHO_WINDOW)                  # inclusive tail window
    return any(s[i:i + _ECHO_WINDOW] in text for i in starts)


#: JSON-mode authoring budget. The author emits a full CompetenceLadder /
#: GamingArchetypeSet JSON (nested rungs/groups) + adaptive thinking — generous so a
#: complete object fits (a truncated one fails to parse → recorded skip, never silent).
_AUTHOR_MAX_TOKENS = 12000


def _iter_json_objs(text: str):
    """Yield EVERY top-level JSON object in `text`, left-to-right, regardless of ```json
    fences or surrounding prose (raw_decode skips the fence markers and any non-JSON
    brace). The caller validates each and keeps the first that fits the schema — so a
    malformed or decoy object that appears BEFORE the genuine one does not shadow it
    (round-4 review hardening)."""
    if not text or not text.strip():
        return
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        start = text.find("{", i)
        if start < 0:
            break
        try:
            obj, end = dec.raw_decode(text[start:])
        except (ValueError, RecursionError):
            # ValueError = a non-JSON brace; RecursionError = JSON nested beyond
            # CPython's limit (raw_decode raises it, NOT a ValueError) — both mean
            # "not a usable object here", so skip this brace and keep scanning. Keeps
            # the helper self-contained (never raises) independent of caller wrapping.
            i = start + 1
            continue
        if isinstance(obj, dict):
            yield obj
        i = start + max(end, 1)


def _extract_json_obj(text: str) -> dict:
    """Extract the FIRST top-level JSON object from a model completion — tolerant of a
    ```json fence or surrounding prose. Raises ValueError (incl its subclass
    json.JSONDecodeError) if none parses; the caller records a skip, so a
    malformed/truncated author NEVER grants. (See `_iter_json_objs` for the all-candidate
    scan `_author_structured` uses to pick the first SCHEMA-VALID object.)"""
    for obj in _iter_json_objs(text):
        return obj
    raise ValueError("no JSON object in author response")


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
    # Try EVERY candidate JSON object and keep the first that validates AND is NON-TRIVIAL
    # — so a decoy/prose block before the genuine output doesn't shadow it. CRITICAL: the
    # CompetenceLadder / GamingArchetypeSet schemas have all-default fields (rungs=[],
    # archetypes=[]), so a stray `{}` / `{"x":1}` VALIDATES with an EMPTY content list. If
    # such an empty object shadowed the real one, the gaming gate would see 0 archetypes
    # → ran=False → inconclusive → fail OPEN (a gameable metric grants). So an empty
    # content list is REJECTED here (keep scanning); if no candidate is non-empty we raise
    # → the caller records a skip (fail-SAFE: inconclusive, never a silent empty accept).
    last_err: Optional[Exception] = None
    for obj in _iter_json_objs(text):
        try:
            result = schema.model_validate(obj)
        except Exception as e:  # noqa: BLE001 — try the next candidate
            last_err = e
            continue
        items = getattr(result, "archetypes", None)
        if items is None:
            items = getattr(result, "rungs", None)
        if items is not None and len(items) == 0:
            last_err = ValueError(
                "structured object has no archetypes/rungs (empty shadow) — skipped")
            continue
        return result
    if last_err is not None:
        raise last_err
    raise ValueError("no JSON object in author response")


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


def _gameable_score(s: float) -> float:
    """§round-7 FALSE-GRANT fix: in the adversarial GAMING gate, a non-finite score on a
    gaming probe must be treated as MAXIMALLY gaming (fail-CLOSED), the OPPOSITE direction
    from the calibration sites (where high-on-a-hack is bad, not good). `NaN > worst` is
    False, so a NaN-scoring hack would silently drop out of `worst` and ESCAPE the
    gameability denial → a false grant. Coerce non-finite → 1.0 (≥ any ceiling) so a hack
    the metric scores undefined is denied, not waved through. Also keeps meta.json JSON
    valid (no bare NaN persisted)."""
    return s if np.isfinite(s) else 1.0


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

#: §round-13 the GENERAL (goal-agnostic) gaming channels every task-derived grant
#: must defend: a CONFOUND-only proxy that reads posture/height or one-way pelvis
#: depth without performing the goal.
_GENERAL_SCORED_CHANNELS = ("posture", "completion")
#: A goal whose competent END-STATE is on the floor (lie/lay down, sleep, rest,
#: collapse, prone) — for these `collapse_and_stay_down` is ON-goal, so it must NOT
#: be used as a required-loser (would false-deny a legit descend metric).
_TERMINAL_DOWN_TOKENS = ("lie", "lay", "lying", "laying", "sleep", "rest",
                         "resting", "collapse", "prone", "supine")
#: …unless the goal also says it returns/rises (then the down state is transient).
_RETURN_UP_TOKENS = ("up", "rise", "rises", "rising", "stand", "standing",
                     "return", "returns", "back", "recover", "straighten", "tall")
#: A goal whose competent behavior IS standing upright and still (balance / hold a
#: stance) — for these `do_nothing_upright` (and the idle `jitter`) are ON-goal, so they
#: must NOT be used as required-losers (they would false-deny a balance metric whose
#: competent top-rung renders identically to a still upright stance; the synthesizer can't
#: render single-leg contact, so a one-legged balance is physically indistinguishable from
#: do_nothing_upright). §round-16: a balance goal is often phrased WITHOUT "balance" ("stand
#: on one leg", "remain upright on one foot", "do not fall over") → these posture/anti-fall
#: cues + phrases are added, behind the active-token guard so an active goal is never
#: misread. Bare "stand"/"hold" stay OUT (they appear in fold goals like "stand back up").
_STATIC_HOLD_TOKENS = ("balance", "balanced", "balancing", "still", "motionless",
                       "stationary", "immobile", "upright", "stance", "equilibrium",
                       "poise", "poised")
#: balance/anti-fall PHRASES (substring match — single-leg contact the synthesizer can't
#: render, or an explicit don't-fall objective).
_STATIC_HOLD_PHRASES = ("one leg", "one foot", "single leg", "single-leg", "one-legged",
                        "do not fall", "don't fall", "dont fall", "without falling",
                        "stay upright", "remain upright", "keep your balance",
                        "keep balance", "hold a stance", "hold still")
#: §round-15/16: an ACTIVE-motion or LOCOMOTION verb forces static_hold=False regardless of
#: any stillness adverb — "give a steady wave" / "stay still then dash forward" are ACTIVE
#: (the stillness is an incidental modifier, not the objective), so the posture losers must
#: still apply. False-classifying an active goal as static-hold DROPS the posture defense →
#: FALSE GRANT (the dangerous direction), so this list is kept broad and the classifier
#: BIASES toward False when in doubt.
_ACTIVE_MOTION_TOKENS = (
    "wave", "reach", "raise", "lower", "swing", "step", "walk", "run", "march",
    "kick", "punch", "gesture", "lift", "turn", "twist", "bend", "extend", "throw",
    "fold", "squat", "bow", "crouch", "touch", "nod", "shake", "clap", "shrug",
    "jump", "hop", "lunge", "stomp", "stride", "gait", "rotate", "flex", "curl",
    # §round-16 locomotion/whole-body verbs (a locomotion goal with an incidental
    # stillness adverb must NOT be read as static-hold → idle-upright proxy false grant):
    "dash", "sprint", "crawl", "leap", "slide", "shuffle", "jog", "trot", "gallop",
    "skip", "climb", "roll", "spin", "pivot", "push", "pull", "drag", "carry",
    "scoot", "sidestep", "backpedal", "pirouette", "dance", "strafe", "sway")
#: directional-travel words — a goal that travels is never a static hold (defense-in-depth).
_DIRECTIONAL_TOKENS = ("forward", "forwards", "backward", "backwards", "ahead", "behind",
                       "left", "right", "sideways", "laterally", "across")


def _goal_is_terminal_down(behavior_goal: str) -> bool:
    """True iff the goal's competent end-state is DOWN (lie/rest, no return) — then a
    collapse-and-stay policy is ON-goal and must not be used as a required-loser."""
    toks = set(re.findall(r"[a-z]+", (behavior_goal or "").lower()))
    return bool(toks & set(_TERMINAL_DOWN_TOKENS)) and not (toks & set(_RETURN_UP_TOKENS))


def _goal_is_static_hold(behavior_goal: str) -> bool:
    """KEYWORD FALLBACK (used only when the authored ladder is unavailable — e.g. the gate
    is exercised directly in a test). The AUTHORITATIVE signal is the blind ladder's own
    top-rung posture (`_ladder_posture`), goal-text-independent and anti-collusion-safe; a
    keyword classifier over free-text is inherently brittle (rounds 13/15/16/17 each found
    a token gap). True iff the goal's competent behavior is a still upright hold.

    BIASED TOWARD FALSE (the safe direction): a false True drops the posture defense → a
    FALSE GRANT, whereas a false False merely keeps a loser → observe-only. So ANY
    active-motion/locomotion verb OR a directional-travel cue forces False, and True
    requires POSITIVE balance evidence (a static-hold phrase or a balance-DOMINANT token —
    weak modifiers like "upright"/"stance"/"steady" are NOT sufficient alone, since they
    appear in active goals like "salute while staying upright")."""
    g = (behavior_goal or "").lower()
    toks = set(re.findall(r"[a-z]+", g))
    stems = {t.rstrip("s") for t in toks} | {t[:-3] for t in toks if t.endswith("ing")} \
        | {t[:-2] for t in toks if t.endswith("ed")}
    active = {a.rstrip("s") for a in _ACTIVE_MOTION_TOKENS}
    if (toks & set(_ACTIVE_MOTION_TOKENS)) or (stems & active):   # incl. inflected forms
        return False
    if toks & set(_DIRECTIONAL_TOKENS):
        return False
    if any(p in g for p in _STATIC_HOLD_PHRASES):
        return True
    return bool(toks & set(_STATIC_HOLD_TOKENS))


def _scalar_end(v: Any) -> float:
    """The END value of a MotionSpec Scalar (a float, or a [start, end] ramp)."""
    try:
        if isinstance(v, (list, tuple)):
            return float(v[-1]) if v else 0.0
        return float(v)
    except Exception:  # noqa: BLE001
        return 0.0


def _scalar_start(v: Any) -> float:
    """The START value of a MotionSpec Scalar (a float, or a [start, end] ramp)."""
    try:
        if isinstance(v, (list, tuple)):
            return float(v[0]) if v else 0.0
        return float(v)
    except Exception:  # noqa: BLE001
        return 0.0


def _spec_is_static_hold(spec: Any) -> bool:
    """True iff a competence-ladder TOP rung describes a STILL UPRIGHT hold — high
    uprightness, no pelvis fold, no travel/hops, and no commanded joint motion. This reads
    the blind author's own notion of competence (anti-collusion: the author never sees the
    metric), so it decides whether do_nothing_upright is ON-goal WITHOUT brittle goal-text
    keywords."""
    try:
        if _scalar_end(getattr(spec, "uprightness", 1.0)) < 0.8:
            return False
        # §round-19: base_height_m is a motion/posture channel too (it was the one
        # MotionSpec motion field this detector did not read). A STILL UPRIGHT hold matches
        # do_nothing_upright's STANDING posture (base_height_m≈0.7). A base_height RAMP is
        # commanded VERTICAL MOTION (rise/descend), and a held NON-nominal height (squat) is
        # a DIFFERENT posture where standing-still is OFF-goal — both must DROP the
        # static-hold classification so do_nothing/jitter stay as losers. Defense-in-depth:
        # today the prepended fallen anchor (z≈0.5) incidentally breaks a pure-height
        # confound, but the firewall must not depend on that. SAFE vs false-reject: a genuine
        # balance metric (which scores do_nothing HIGH) only authors a NOMINAL-height hold
        # (≥0.55, no ramp) → still static → do_nothing still dropped; a genuine ramp/squat
        # metric scores do_nothing LOW (it rewards the motion/low posture) → keeping the
        # loser cannot deny it.
        bh = getattr(spec, "base_height_m", 0.7)
        if _scalar_start(bh) < 0.55 or _scalar_end(bh) < 0.55:
            return False
        if abs(_scalar_end(bh) - _scalar_start(bh)) > 0.05:
            return False
        if abs(float(getattr(spec, "fold_depth_m", 0.0) or 0.0)) > 0.05:
            return False
        if abs(float(getattr(spec, "tremor", 0.0) or 0.0)) > 0.1:
            return False   # §round-18: tremor is a whole-body motion channel (NOT per-group)
        if abs(float(getattr(spec, "noise", 0.0) or 0.0)) > 0.02:
            return False   # §round-18: noise (gaussian on joint_vel) is whole-body motion too
        if abs(float(getattr(spec, "forward_speed_mps", 0.0) or 0.0)) > 0.1:
            return False
        if abs(float(getattr(spec, "lateral_speed_mps", 0.0) or 0.0)) > 0.1:
            return False
        if int(getattr(spec, "hop_count", 0) or 0) > 0 and float(getattr(spec, "hop_height_m", 0.0) or 0.0) > 0.05:
            return False
        # §round-18: a STILL hold has NO commanded joint deviation. ANY group that produces
        # a non-trivial joint motion (an oscillate/burst/fold group with a real amplitude/
        # peak/burst) OR a held distinctive posture (a 'hold' offset) makes the rung ACTIVE,
        # NOT static — even a SMALL/subtle gesture (amplitude ≤ 0.1). The old amplitude>0.1
        # threshold mis-read a subtle wave as a hold → dropped the velocity defense → a
        # velocity-confound proxy false-granted. A genuine balance/hold top has no such group.
        for gr in (getattr(spec, "groups", []) or []):
            amp = abs(float(getattr(gr, "amplitude_rad", 0.0) or 0.0))
            peak = abs(float(getattr(gr, "peak_radps", 0.0) or 0.0))
            burst = int(getattr(gr, "burst_count", 0) or 0)
            off = abs(float(getattr(gr, "offset_rad", 0.0) or 0.0))
            if amp > 1e-3 or peak > 1e-3 or burst > 0 or off > 1e-2:
                return False
        return True
    except Exception:  # noqa: BLE001 — unparseable spec → not static-hold (keep losers)
        return False


def _spec_is_terminal_down(spec: Any) -> bool:
    """True iff a competence-ladder TOP rung describes a DOWN end-state (lie/rest) —
    non-upright or near-floor at the top, so collapse_and_stay_down is ON-goal."""
    try:
        return (_scalar_end(getattr(spec, "uprightness", 1.0)) <= 0.3
                or _scalar_end(getattr(spec, "base_height_m", 0.7)) <= 0.35)
    except Exception:  # noqa: BLE001
        return False


def _ladder_posture(top_specs: list) -> tuple[Optional[bool], Optional[bool]]:
    """Aggregate the AUTHORED top-rung specs of the valid ladders into (static_hold,
    terminal_down) by STRICT MAJORITY. Returns (None, None) when there is no usable
    evidence (caller falls back to the goal-text keyword classifier). Ties → False (the
    safe direction: keep the losers)."""
    specs = [s for s in (top_specs or []) if s is not None]
    if not specs:
        return None, None
    n = len(specs)
    sh = sum(1 for s in specs if _spec_is_static_hold(s))
    td = sum(1 for s in specs if _spec_is_terminal_down(s))
    return (sh * 2 > n), (td * 2 > n)


def general_required_losers(
    joint_names: list[str], behavior_goal: str = "",
    *, static_hold: Optional[bool] = None, terminal_down: Optional[bool] = None,
) -> list[dict]:
    """§round-13 FALSE-GRANT fix: DETERMINISTIC, goal-AGNOSTIC gaming probes that a
    genuine metric of ANY active goal must score LOW, but a CONFOUND-only proxy
    scores HIGH — so the firewall no longer relies on a blind LLM author happening
    to propose the catching archetype (the depth-only / posture-only proxies that
    false-granted on fold/gesture goals). Each is `{name, channel, arrays, behavior,
    meta}`, rendered offline via the Ship-51 synthesizer; run REGARDLESS of family or
    flags. Never raises.

      * do_nothing_upright — near-still + fully upright at nominal height. Any ACTIVE
        goal (fold/gesture/locomotion/kick) is unperformed → a real metric scores ~0; a
        posture/height proxy scores it MAX. (catches the posture proxy)
      * jitter_in_place — a SMALL idle high-frequency twitch: catches a proxy that
        rewards generic joint VELOCITY/"some motion" (its 15 Hz gives high velocity at a
        deliberately small ROM, so an honest small-amplitude gesture metric is NOT
        false-rejected).
      * collapse_and_stay_down — a deep pelvis dip that NEVER returns upright → catches a
        dip-DEPTH-only proxy that ignores the 'stand back up' half.

    §round-15 carve-outs (so the loser set is NEVER empty and every loser is OFF-goal):
      - do_nothing_upright/jitter are ON-goal ONLY for a still-UPRIGHT (balance) goal, i.e.
        static-hold AND NOT terminal-down → dropped only then. For a terminal-down "lie
        still" goal a still-UPRIGHT policy is OFF-goal (wrong height) so they are KEPT.
      - collapse is ON-goal for a terminal-DOWN goal (lie/rest) → dropped only then.
      So: balance→{collapse}; lie-down/lie-still→{do_nothing,jitter}; active→all three.

    §round-17: `static_hold`/`terminal_down` are normally derived from the blind AUTHORED
    ladder's top-rung posture (`_ladder_posture`) — goal-text-independent and
    anti-collusion-safe — which replaced the brittle goal-keyword classifier (it had a new
    token gap every round). When not supplied they fall back to the keyword classifier (for
    direct/test callers without a ladder).

    A raise while scoring a loser is handled as score 0.0 (counted) in the gate, so an
    honest onset/phase metric that indexes an empty selection on a degenerate loser is NOT
    false-rejected, and a real reward-the-degenerate proxy (which returns a HIGH score, not
    a raise) is still caught — see the gate's required-loser loop for the (compositional)
    soundness argument.
    """
    from sculptor.eval.ladder_synth import MotionSpec, render_rung

    def _pack(name: str, channel: str, spec: Any, idx: int) -> dict:
        arrays, behavior, meta = render_rung(spec, joint_names, rung_index=idx)
        return {"name": name, "channel": channel, "arrays": arrays,
                "behavior": behavior, "meta": dict(meta)}

    sh = _goal_is_static_hold(behavior_goal) if static_hold is None else static_hold
    td = _goal_is_terminal_down(behavior_goal) if terminal_down is None else terminal_down
    losers: list[dict] = []
    if not (sh and not td):   # still-upright losers are on-goal ONLY for a balance goal
        # do_nothing is PERFECTLY still + upright: a posture/height proxy AND a stillness
        # proxy both score it MAX, while any active goal scores it ~0.
        losers.append(_pack("do_nothing_upright", "posture",
                            MotionSpec(uprightness=1.0, base_height_m=0.7), 600))
        # a SMALL twitch (tremor 0.04 ≈ 0.07 rad ROM, peak vel ~3.5 rad/s): enough joint
        # VELOCITY at 15 Hz to trip a "rewards any motion" proxy, ROM far below any
        # plausible gesture target so an honest small-amplitude gesture metric is not denied.
        losers.append(_pack("jitter_in_place", "posture",
                            MotionSpec(uprightness=1.0, base_height_m=0.7,
                                       tremor=0.04), 601))
    if not td:
        losers.append(_pack("collapse_and_stay_down", "completion",
                            MotionSpec(uprightness=0.0, base_height_m=[0.7, 0.1]), 602))
    return losers


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
    author: bool = True,
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
        except Exception as e:  # noqa: BLE001
            # §round-15/16 RAISE→0.0, COUNTED: a raise scores the loser 0.0 and counts it
            # (losers_scored += 1, so ran=True — no fail-open via an empty set). It is
            # neither "evasion" (round-14's fail-closed → false-rejected honest onset/phase
            # metrics that index an empty selection on a degenerate rollout) nor "no
            # evidence" (round-13's skip → fail-open via ran=False).
            #
            # SOUNDNESS (§round-16 — it is NOT a per-loser gate⟺runtime equivalence): the
            # synth losers are byte-clean (all E envs identical, exact constants) while a
            # real rollout varies per-env, so a metric CAN raise here yet score the real
            # degenerate policy high at runtime. Safety is COMPOSITIONAL: the same
            # synth-clean signature that raises here also fires on render_ladder's
            # always-clean _ANCHOR rung AND the prompt-mandated clean competent top rungs →
            # it collapses the metric's rho/separation so it FAILS the ladder gate and never
            # reaches this gate. (And a metric that genuinely REWARDS a degenerate policy
            # returns a HIGH score here, not a raise → caught.) 0.0 never bumps `worst`, so
            # an honest metric that merely IndexErrors on a degenerate loser is not denied.
            losers_scored += 1
            rec["required_losers"].append(
                {"name": name, "channel": loser.get("channel"), "score": 0.0,
                 "note": f"unscorable (→0.0): {type(e).__name__}"})
            continue
        s = _gameable_score(s)   # §round-7: a NaN/inf hack score → GAMEABLE (fail-closed)
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
    if not author:
        # §round-13: deterministic-losers-only mode (the always-on default firewall).
        # The LLM breadth pass is opt-in (adversarial=True); skipping it spends no call
        # and never fails open — the required_losers above carry the verdict.
        author_reason = "adversarial: LLM breadth pass not requested (deterministic losers only)"
    elif "def compute_spec" in payload_text or (
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
        if _echoes_source(metric_src, gset_text):
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
            s = _gameable_score(s)   # §round-7: a NaN/inf hack score → GAMEABLE (fail-closed)
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
    gameable = ((not np.isfinite(worst))   # §round-7: non-finite worst → fail-closed
                or (worst >= rel_ceil * float(competent_ref)) or (worst >= abs_ceil))
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
    valid_top_specs: list = []   # §round-17: authored top-rung specs of valid ladders
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
        if _echoes_source(metric_src, ladder_text):
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
                s = float(gen_fn(arrays, behavior, meta).get("spec_score", 0.0))
            except Exception:  # noqa: BLE001 — crash on a rung = 0 (penalize)
                s = 0.0
            # §round-6 FALSE-GRANT fix: a NaN/inf spec_score must be PENALIZED to 0.0,
            # not propagated — a NaN ranks as max in _midrank (→ spurious rho≈1.0), and
            # the std-floor + separation gates are `nan < x` == False (bypassed), so a
            # NaN top-rung would FALSE-GRANT steer-rights. Coerce here (single source).
            gen_scores.append(s if np.isfinite(s) else 0.0)
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
        valid_top_specs.append(rungs[-1] if rungs else None)   # §round-17 ladder posture
        # ABSOLUTE-SEPARATION anchor indicts the METRIC (counts as disagreement).
        # §round-6 defense-in-depth: a non-finite separation must FAIL the gate (a bare
        # `nan < min` is False and would pass) — gen_scores are coerced finite above, so
        # this is belt-and-suspenders.
        if not np.isfinite(separation) or separation < _TD_SEPARATION_MIN:
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

    # §round-13 FALSE-GRANT fix: the gate now runs whenever base_ok (was opt-in
    # `adversarial` only → a posture/depth CONFOUND-only proxy false-granted in the
    # default path, because the blind ladders co-vary the confound with rung and the
    # only goal-aware defense was off). The DETERMINISTIC goal-blind losers
    # (general_required_losers) run ALWAYS and carry the verdict — they never fail
    # open. The metric's competent reference is its best top-rung score across valid
    # sources (most charitable, hardest to false-deny).
    adv = None
    if base_ok:
        try:
            competent_ref = max(
                (s["gen_scores"][-1] for s in sources
                 if s.get("ladder_ok") and s.get("gen_scores")), default=0.0)
            from sculptor.eval.metric_validate import resolve_behavior_family
            fam = resolve_behavior_family(behavior_goal, robot_hint)
            if fam == "kick":
                # A KICK has DEDICATED direction/completion/amplitude losers (WITH
                # foot_pos_b); the general posture/depth losers mis-fire on an
                # intensity-based kick metric, so they are NOT used here. The kick
                # losers stay opt-in (a canonical kick routes to the built-in path).
                req_losers = (kick_required_losers(names, behavior_goal, robot_hint)
                              if adversarial_required_losers else None)
                sc = list(_KICK_SCORED_CHANNELS) if adversarial_required_losers else None
            else:
                # §round-13 FALSE-GRANT fix: the general goal-blind losers (do-nothing /
                # jitter / collapse-and-stay) run ALWAYS for a novel (family=None) fold/
                # posture/gesture grant — a real metric scores them ~0; a posture/depth
                # CONFOUND-only proxy scores one at/above the ceiling → gameable → denied.
                # §round-17: whether the still-upright losers are ON-goal (a balance/lie
                # task) is decided from the blind AUTHORED ladder's top-rung posture, NOT a
                # brittle goal-keyword classifier (which had a new token gap every round).
                ladder_sh, ladder_td = _ladder_posture(valid_top_specs)
                req_losers = general_required_losers(
                    names, behavior_goal, static_hold=ladder_sh, terminal_down=ladder_td)
                sc = list(_GENERAL_SCORED_CHANNELS)
            # The LLM breadth pass is opt-in (`adversarial`); the deterministic losers
            # always run, so skipping it never fails the firewall open.
            adv = adversarial_archetype_gate(
                gen_fn, roles, names, competent_ref, client=client, model=model,
                base_payload=base_payload, metric_src=metric_src,
                n_archetypes=adversarial_n,
                required_losers=req_losers, scored_channels=sc,
                author=adversarial)
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
