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
#: §round-21: a STRICT per-loser FLOOR (fraction of competence) for the pure-idle probes
#: (do_nothing_upright) that perform NONE of the goal. The upper ceiling (≈0.5) left an open
#: [0,0.5) band an additive uprightness-gated floor could farm; a probe that does literally
#: nothing must score NEAR ZERO. Only attached to losers that carry a `max_frac` key — the
#: motion-bearing probes (jitter) keep the ceiling so an honest velocity goal isn't denied.
_LOSER_FLOOR_FRAC = 0.15
#: §round-26: the PEAK discriminator threshold. A metric is farming the mean-velocity floor if it
#: scores jitter_in_place (low peak) ≥ this fraction of velocity_peak_ref (high peak, same tiny ROM)
#: — i.e. it is INSENSITIVE to a >2× peak-velocity difference (a flat velocity floor), AND jitter is
#: above the idle floor. A genuine peak/ROM metric scores velocity_peak_ref well above jitter.
#: §round-27 [HIGH FALSE GRANT] recalibration: 0.85 was FAR too loose. An additive confound
#: `up·(a·floor(mean|jv|) + b·rom)` where `floor` is ANY sub-saturating concave map (1−exp with
#: k≥0.6, tanh, sqrt, …) keeps jitter/peak_ref BELOW 0.85 while still farming 32–45% of competence
#: on the OFF-GOAL idle jitter probe (the rom term ranks the ladder → passes the base gate AND
#: lifts competence so the 0.5 abs-ceiling binds, leaving the whole [0,0.5)×[ratio,0.85) window
#: unguarded — 6 reproduced confounds A1/A2/A3/A4/A5/D1). The round-26 claim that the ratio is
#: "robust to tuning" held only for FLOOR_v, NOT for the saturation FUNCTION/rate. The EMPIRICAL
#: separation is clean and threshold-able: the honest fixtures top out at ratio 0.531 (GOOD_KICK
#: and a generous-/8-scale peak metric — both legitimately pay the idle twitch ~0.38–0.52 because
#: an idle twitch HAS peak velocity, which is ACCEPTED for a peak metric), while every reproduced
#: saturating-floor confound floors at 0.615. 0.57 sits in that gap. The precondition
#: (jitter ≥ _LOSER_FLOOR_FRAC·comp, below) protects LOW-farm honest metrics (rom-gated → jitter
#: ~0.03); the ratio protects HIGH-farm-but-PEAK-SENSITIVE honest metrics (ratio < 0.531). A
#: confound tuned to ratio < 0.57 must become peak-sensitive enough that it CONVERGES to the honest
#: generous-peak-metric form — at which point its idle farm is ≤ what the honest reference fixtures
#: themselves exhibit, so it is no more gameable than the accepted baseline. RESIDUAL (honest): this
#: discriminator, like the free-text posture classifier, sits over an UNBOUNDED functional surface —
#: it cannot be PROVEN complete; it bounds the idle-farm residual to the honest-fixture level, not to
#: zero. The threshold is coupled to the fixed probe amplitudes (jitter tremor 0.04 vs peak_ref 0.10)
#: and GOOD_KICK's 0.531 — re-derive if either changes.
#: §round-31 [CONFIRMED STRUCTURAL LIMIT — do NOT chase with another ratio tweak]: the ratio is
#: DECOUPLABLE. A confound `up·(a·sigmoid_peak_gate(peak) + clip(mean|jv|, 0, cap))` ranks the
#: ladder via the peak gate (low ratio, like an honest peak metric) while a HARD-CAPPED (clipped-
#: affine, NOT concave) mean term adds the SAME constant `cap` to BOTH the low-peak jitter and the
#: high-peak velocity_peak_ref probe (both saturate the cap), so the ratio is UNMOVED by the farm
#: and stays < 0.57 while the cap farms idle jitter. CRUCIALLY there is NO velocity-floor threshold
#: that catches this: the confound's jitter-absolute (0.28) AND ratio (0.41) are BOTH BELOW honest
#: GOOD_KICK's (0.354, 0.531) — GOOD_KICK pays an idle knee-twitch MORE in absolute terms — so any
#: threshold catching the confound also false-rejects the honest reference. This is the velocity-
#: floor surface's structural ceiling (4th bypass after knife-edge / fail-open / channel-decouple):
#: the ratio bounds the idle-farm residual to ~the honest-fixture level (~0.52 absolute), not zero,
#: exactly like the keyword classifier + AST denylist. The DURABLE close is the min-composition law
#: (an additive/decoupled `gate + farm` SUM violates `completion_gate·min(channels)`) + goal-joint
#: scoping — NOT a probe/threshold iteration. (The round-31 jitter-RAISE fail-open IS a clean bug,
#: fixed in the gate; B1's marginal cap-farm is the documented residual, not patched.)
_VEL_FLOOR_RATIO = 0.57


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
#: §round-28 [HIGH FALSE GRANT] fix: a goal like "collapse then JUMP", "lie down then SPRING
#: UPWARD", "rest low then BOUND off the ground" returns up via a JUMP-family verb that the
#: minimal set missed (and "upward" tokenizes whole — NOT the standalone "up"), so
#: _goal_is_terminal_down stayed True, the round-27 ladder_td guard did NOT fire, and a
#: drop-to-floor-and-stay confound GRANTED on a blind-author-mis-rendered descent ladder.
#: Broadening this list is the SAFE direction — a return-up token only ever flips terminal_down
#: to False, which KEEPS collapse_and_stay_down (an observe-only false-reject at worst on a
#: genuine lie/rest goal that happens to name a jump verb), NEVER drops a loser → never a
#: false-grant. (Symmetric to the round-24 lesson that broadening the ACTIVE list is the safe
#: direction for static_hold.) The durable robust fix remains goal-joint scoping.
#: §round-32 [MEDIUM FALSE REJECT] fix: the body-part / posture NOUNS "back"/"feet"/"overhead"
#: were REMOVED from this set. They were added (rounds 28-30) as return-to-standing CUES, but
#: they are AMBIGUOUS — "lie on your back and rest", "lie with your feet up and rest", "lie down
#: with arms overhead and rest" are GENUINELY terminal-down goals that contain them, so they
#: false-flipped _goal_is_terminal_down → False → the round-27 ladder_td backstop OVERRODE the
#: correct down-ending LADDER signal → collapse_and_stay_down was KEPT → an HONEST lie-rest metric
#: that legitimately scores a collapsed policy ≥ the ceiling was firewall-DENIED (reproduced 3/3 on
#: the commonest supine phrasings). This contradicted the "broadening is observe-only at worst"
#: claim: keeping the loser on a terminal goal whose honest metric scores it high is a HARD deny.
#: Every "to your feet" / "get back up" return-up TEST goal also carries a rising VERB (rebound/
#: scramble/kip/pike/stand/up), which still classifies it non-terminal, so removal is regression-
#: safe; and the live path's authoritative ladder signal protects a genuine get-up goal (its ladder
#: ends UPRIGHT → ladder_td=False regardless of keywords). Only unambiguous rising VERBS/posture
#: words belong here, never bare body-part nouns.
_RETURN_UP_TOKENS = ("up", "rise", "rises", "rising", "stand", "standing",
                     "return", "returns", "recover", "straighten", "tall",
                     # §round-28: the jump / leap / ascend family (a goal returning up via a
                     # ballistic/rising verb, incl. the "up" compounds that tokenize whole):
                     "jump", "jumps", "jumping", "leap", "leaps", "leaping", "spring",
                     "springs", "springing", "hop", "hops", "hopping", "bound", "bounds",
                     "bounding", "vault", "vaults", "vaulting", "explode", "explodes",
                     "exploding", "launch", "launches", "launching", "soar", "soars",
                     "soaring", "ascend", "ascends", "ascending", "climb", "climbs",
                     "climbing", "lift", "lifts", "lifting", "upward", "upwards",
                     "skyward", "airborne", "aloft", "erect", "vertical",
                     # §round-29: the return-to-feet / RIGHTING family (a goal that ends by
                     # getting back UP off the floor via a non-ballistic righting verb the
                     # round-28 jump family missed: "rebound to your feet", "heave yourself off
                     # the ground", "scramble to your feet", "kip onto your feet", "peel yourself
                     # off the floor", "raise your body off the floor"). Still the SAFE direction
                     # (only ever KEEPS collapse_and_stay_down). Deliberately NOT generic verbs
                     # (come/get/move) that appear in genuine terminal goals like "come to rest".
                     "rebound", "rebounds", "rebounding", "raise", "raises", "raising",
                     "heave", "heaves", "heaving", "scramble", "scrambles", "scrambling",
                     "kip", "kips", "kipping", "peel", "peels", "peeling", "righting",
                     "arise", "arises", "arising", "clamber", "clambers", "clambering",
                     "unfold", "unfolds", "unfolding", "elevate", "elevates", "elevating",
                     "feet", "upturn", "upend",
                     # §round-30: the lift-SELF-up / gymnastics-rise family (a returns-up goal
                     # ending by hauling the body off the floor with a verb the round-28 jump and
                     # round-29 righting families missed — "hoist your torso upright", "haul
                     # yourself off the ground", "pike up", "bridge up", "muscle up"). Notably
                     # "heave" was closed in round-29 while its synonyms hoist/haul were not. Still
                     # the SAFE direction (only ever KEEPS collapse_and_stay_down).
                     "hoist", "hoists", "hoisting", "haul", "hauls", "hauling", "wrench",
                     "wrenches", "wrenching", "jolt", "jolts", "jolting", "propel", "propels",
                     "propelling", "pike", "pikes", "piking", "bridge", "bridges", "bridging",
                     "lever", "levers", "levering", "jackknife", "jackknifes", "jackknifing",
                     "muscle", "muscles", "muscling", "pop", "pops", "popping",
                     # §round-32 [HIGH FALSE GRANT] fix: the SIT-UP / return-to-vertical family the
                     # jump/righting/lift-self-up families missed — "collapse to the floor then sit
                     # upright", "lie prone then sit your torso to upright". "upright" is the clean,
                     # unambiguous vertical cue (a terminal-down goal is by definition NOT upright,
                     # so it rarely collides — unlike the body-part nouns removed above); situp/
                     # situps are unambiguous one-word gymnastics-rise terms. NOT bare "sit" (it is
                     # ambiguous — "sit DOWN and rest" is terminal — and would re-introduce the
                     # body-part-noun false-reject for the very common seated-rest goals).
                     "upright", "situp", "situps",
                     # §round-33 [HIGH FALSE GRANT] fix: removing the body-part noun "back" in
                     # round-32 opened "collapse to the floor then ARCH your back off the ground" /
                     # "arch your back off the floor" (a genuine rise-off-the-floor bridge/backbend
                     # that now carried NO listed rising token) → terminal_down=True → collapse_and_
                     # stay_down dropped → a low-and-still confound granted on a mis-rendered descent
                     # ladder. "arch" is the actual rising VERB in these phrasings (a back-arch lifts
                     # the body off the ground); unlike the noun "back" it is unambiguous (a terminal
                     # lie/rest goal does not "arch"). SAFE direction (only KEEPS the loser).
                     "arch", "arches", "arching")
#: A goal whose competent behavior IS standing upright and still (balance / hold a
#: stance) — for these `do_nothing_upright` (and the idle `jitter`) are ON-goal, so they
#: must NOT be used as required-losers (they would false-deny a balance metric whose
#: competent top-rung renders identically to a still upright stance; the synthesizer can't
#: render single-leg contact, so a one-legged balance is physically indistinguishable from
#: do_nothing_upright). §round-16: a balance goal is often phrased WITHOUT "balance" ("stand
#: on one leg", "remain upright on one foot", "do not fall over") → these posture/anti-fall
#: cues + phrases are added, behind the active-token guard so an active goal is never
#: misread. Bare "stand"/"hold" stay OUT (they appear in fold goals like "stand back up").
#: §round-24: REVERTED to the stable minimal set. The round-22/23 broadening (freeze/frozen/
#: rigid/statue/flamingo + "center of mass"/"stay on your feet"/"t-pose"/…) was a MISTAKE —
#: a balance keyword conjoined with an ACTIVE objective whose verb is NOT on the active list
#: ("shift your CENTER OF MASS", "play the STATUE game", "salute and STAY ON YOUR FEET") then
#: drops the posture losers → FALSE GRANT. Broadening positive balance cues is NOT
#: accept-rate-safe (it can false-GRANT); only broadening the ACTIVE-verb list is safe (it
#: only ever KEEPS the losers). The residual "active-verb-not-on-the-list + balance keyword"
#: false-grant is a PRE-EXISTING limitation of any free-text classifier — the DURABLE fix is
#: goal-joint scoping (REQUIRED_JOINT_ROLES), a separate increment.
_STATIC_HOLD_TOKENS = ("balance", "balanced", "balancing", "still", "motionless",
                       "stationary", "immobile", "upright", "stance", "equilibrium",
                       "poise", "poised")
#: balance/anti-fall PHRASES (substring match — single-leg contact the synthesizer can't
#: render, so a one-legged balance is indistinguishable from do_nothing_upright; or an
#: explicit don't-fall objective).
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
    "scoot", "sidestep", "backpedal", "pirouette", "dance", "strafe", "sway",
    # §round-24: gesture/manipulation/whole-body verbs whose ABSENCE let a balance keyword
    # ('salute and balance', 'shift your weight while balancing', 'flap like a flamingo')
    # mis-classify an ACTIVE goal as a static hold → posture confound false-grant. Broadening
    # the ACTIVE list is SAFE — it only ever KEEPS the do_nothing/jitter losers (the safe
    # direction), never drops them, so it cannot false-GRANT. (Deliberately NOT the broad
    # auxiliaries do/act/play/perform — they appear in balance goals like "do not fall over".)
    "salute", "flap", "wiggle", "snap", "scrub", "hammer", "shift", "rock",
    "pump", "clap", "stir", "conduct", "windmill", "shadowbox", "applaud", "curtsy",
    "mimic", "vibrate", "oscillate", "weave", "circle", "pedal", "drum", "knock",
    "poke", "prod", "jab", "chop", "slice", "whisk", "paddle", "row", "wring",
    "wobble", "jiggle", "bounce", "thrust", "twirl", "flail", "swat",
    "box", "point", "mime", "juggle", "dribble", "toss", "tap",
    # §round-25 reproduced gaps (manipulation / sport / dance / fidget gestures):
    "paint", "knead", "sweep", "strum", "solder", "serve", "putt", "waltz", "mop",
    "vacuum", "dust", "sketch", "whittle", "sand", "sew", "carve", "grate", "peel",
    "polish", "iron", "frost", "rake", "stack", "assemble", "screw", "tighten",
    "unscrew", "crank", "ladle", "pour", "scoop", "flip", "saw", "drill", "buff",
    "trace", "vogue", "tango", "salsa", "krump", "twerk", "bowl", "fence", "dunk",
    "spike", "volley", "cast", "fidget", "gesticulate", "bob", "headbang", "shimmy",
    "gyrate")
#: directional-travel words — a goal that travels is never a static hold (defense-in-depth).
_DIRECTIONAL_TOKENS = ("forward", "forwards", "backward", "backwards", "ahead", "behind",
                       "left", "right", "sideways", "laterally", "across")
#: §round-32 [CRITICAL FALSE GRANT] fix: LOCOMOTION verbs — a goal whose competent behavior
#: TRAVELS the base across the ground (walk/run/march/…). For these, the upright-while-traveling
#: probe `walk_away_upright` is ON-goal (dropped); for a STATIONARY goal it is OFF-goal (kept), so
#: an additive SUM that farms the wholly-uncovered horizontal-travel channel (a run-forward policy
#: that performs none of the in-place goal) is caught. KEYWORD FALLBACK only — the authoritative
#: signal is the blind ladder's own commanded travel (`_ladder_travels`), goal-text-independent and
#: anti-collusion-safe. "march in place"/"jog on the spot" are deliberately NOT travel (the
#: stationary qualifier wins) — and the ladder backstops them regardless.
_LOCOMOTION_TOKENS = ("walk", "walks", "walking", "run", "runs", "running", "march",
                      "marches", "marching", "jog", "jogs", "jogging", "sprint", "sprints",
                      "sprinting", "dash", "dashes", "dashing", "stride", "strides",
                      "striding", "crawl", "crawls", "crawling", "gallop", "gallops",
                      "galloping", "trot", "trots", "trotting", "advance", "advances",
                      "advancing", "traverse", "traverses", "traversing", "locomote",
                      "wander", "wanders", "wandering", "travel", "travels", "traveling",
                      "travelling",
                      # §round-33: the LATERAL / varied-gait locomotion family (a goal-verb the
                      # round-32 set missed → the goal-text fallback false-negatived a genuine
                      # locomotion goal). Now only a fallback — the authoritative _ladder_travels
                      # signal is trusted directly (see the calibrate_task_derived backstop).
                      "sidestep", "sidesteps", "sidestepping", "strafe", "strafes", "strafing",
                      "shuffle", "shuffles", "shuffling", "backpedal", "backpedals",
                      "backpedalling", "backpedaling", "scoot", "scoots", "scooting",
                      "scuttle", "scuttles", "scuttling", "pace", "paces", "pacing",
                      "slide", "slides", "sliding", "skip", "skips", "skipping",
                      "scamper", "scampers", "scampering", "amble", "ambles", "ambling")
_STATIONARY_QUALIFIERS = ("in place", "on the spot", "in-place", "without moving",
                          "without traveling", "stay put", "staying put")
_LOCOMOTION_SPEED_MIN = 0.3   # m/s — a ladder rung at/above this commands sustained travel


def _goal_is_terminal_down(behavior_goal: str) -> bool:
    """True iff the goal's competent end-state is DOWN (lie/rest, no return) — then a
    collapse-and-stay policy is ON-goal and must not be used as a required-loser."""
    toks = set(re.findall(r"[a-z]+", (behavior_goal or "").lower()))
    return bool(toks & set(_TERMINAL_DOWN_TOKENS)) and not (toks & set(_RETURN_UP_TOKENS))


def _goal_is_locomotion(behavior_goal: str) -> bool:
    """KEYWORD FALLBACK (used only when no authored ladder is available — the authoritative
    signal is `_ladder_travels`). True iff the goal commands sustained base TRAVEL across the
    ground, so the upright-while-traveling probe `walk_away_upright` is ON-goal (dropped). A
    STATIONARY qualifier ('in place', 'on the spot') vetoes (a 'march in place' does not travel).
    BIASED toward False (the safe direction for the travel probe: a miss KEEPS the loser →
    observe-only, never a gate-weakening false-grant)."""
    g = (behavior_goal or "").lower()
    if any(q in g for q in _STATIONARY_QUALIFIERS):
        return False
    toks = set(re.findall(r"[a-z]+", g))
    return bool(toks & set(_LOCOMOTION_TOKENS))


def _ladder_travels(valid_ladders: Any) -> bool:
    """§round-32: True iff a STRICT MAJORITY of valid ladders command sustained TRAVEL (a rung
    with |forward_speed_mps| or |lateral_speed_mps| ≥ _LOCOMOTION_SPEED_MIN). The blind author's
    competent ladder for a locomotion goal travels; for a stationary goal (kick/wave/fold/balance/
    lie) every rung stays in place. Goal-text-independent + anti-collusion-safe (mirrors
    `_ladder_posture` / `_ladder_has_crouched_rung`). When it travels, `walk_away_upright` is
    on-goal and is dropped; otherwise it is kept so an additive horizontal-travel farm is caught."""
    ladders = list(valid_ladders or [])
    if not ladders:
        return False

    def travels(L: Any) -> bool:
        for rung in (L or []):
            fs = abs(float(getattr(rung, "forward_speed_mps", 0.0) or 0.0))
            ls = abs(float(getattr(rung, "lateral_speed_mps", 0.0) or 0.0))
            if fs >= _LOCOMOTION_SPEED_MIN or ls >= _LOCOMOTION_SPEED_MIN:
                return True
        return False

    return sum(1 for L in ladders if travels(L)) * 2 > len(ladders)


def _goal_is_jump(behavior_goal: str) -> bool:
    """§round-33 KEYWORD FALLBACK (the authoritative signal is `_ladder_hops`). True iff the goal
    commands VERTICAL hopping/jumping in place, so the upright-while-hopping probe `hop_in_place_
    upright` is ON-goal (dropped). BIASED toward False (the safe direction for the hop probe: a miss
    KEEPS the loser → observe-only). NOTE the OVERLAP with _RETURN_UP_TOKENS' jump family is
    intentional: there 'jump' signals a return-up GOAL phase; here it signals a hop COMPETENCE — a
    "collapse then jump up" goal is terminal-down=False (keeps collapse) AND jump=True (drops the hop
    probe, since a competent jumper legitimately hops)."""
    toks = set(re.findall(r"[a-z]+", (behavior_goal or "").lower()))
    return bool(toks & {"jump", "jumps", "jumping", "hop", "hops", "hopping", "leap", "leaps",
                        "leaping", "bound", "bounds", "bounding", "bounce", "bounces", "bouncing",
                        "spring", "springs", "springing", "pogo", "skip", "skips", "skipping",
                        "vault", "vaults", "vaulting"})


def _ladder_hops(valid_ladders: Any) -> bool:
    """§round-33: True iff a STRICT MAJORITY of valid ladders command VERTICAL hopping (a rung with
    hop_height_m > 0 or hop_count > 0). The blind author's competent ladder for a jump/hop goal
    hops; a stationary goal's rungs hold the pelvis flat. Goal-text-independent + anti-collusion-safe
    (mirrors `_ladder_travels`). When it hops, `hop_in_place_upright` is on-goal and dropped;
    otherwise it is kept so an additive VERTICAL-hop (root-z bounce) farm is caught."""
    ladders = list(valid_ladders or [])
    if not ladders:
        return False

    def hops(L: Any) -> bool:
        for rung in (L or []):
            hh = abs(float(getattr(rung, "hop_height_m", 0.0) or 0.0))
            hc = int(getattr(rung, "hop_count", 0) or 0)
            if hh > 1e-6 or hc > 0:
                return True
        return False

    return sum(1 for L in ladders if hops(L)) * 2 > len(ladders)


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
    # §round-23: an ACTIVE-motion verb is checked FIRST (BIAS toward False = the safe direction).
    # An earlier "override phrase" idea (let 'hold still' beat an incidental active verb) was
    # REMOVED — it mis-classified an active SEQUENCE ('hold still THEN dash forward') as a hold,
    # which dropped do_nothing and re-opened a posture false-grant on a mismatched ladder. The
    # cost is that a still-hold goal which incidentally names a motion verb ('freeze mid-stride
    # and hold still') is observe-only — the SAFE direction (a false-reject, never a false-grant).
    if _goal_has_active_motion(behavior_goal):
        return False
    g = (behavior_goal or "").lower()
    if any(p in g for p in _STATIC_HOLD_PHRASES):
        return True
    toks = set(re.findall(r"[a-z]+", g))
    return bool(toks & set(_STATIC_HOLD_TOKENS))


def _goal_has_active_motion(behavior_goal: str) -> bool:
    """True iff the goal NAMES an active-motion/locomotion verb (incl. inflected forms) or a
    directional-travel cue — the POSITIVE signal that the goal is NOT a still hold. §round-22:
    used to VETO a blind-ladder static_hold (keep the do_nothing/jitter posture losers) ONLY
    when the goal is explicitly active, WITHOUT demanding a positive balance keyword. The
    keyword list of static-hold tokens is brittle (most balance phrasings — 'hold a flamingo
    pose', 'freeze in place', 'keep your center of mass over your feet' — miss it), so requiring
    a positive match false-rejected honest balance metrics; checking for a positive ACTIVE verb
    instead keeps the round-21 #6 backstop (an active-gesture goal with a mismatched
    stability-graded ladder) while trusting the authoritative ladder posture for balance goals."""
    g = (behavior_goal or "").lower()
    seq = re.findall(r"[a-z]+", g)
    active_set = set(_ACTIVE_MOTION_TOKENS)
    active_stems = {a.rstrip("s") for a in _ACTIVE_MOTION_TOKENS}
    directional = set(_DIRECTIONAL_TOKENS)
    # §round-25: a NEGATED motion verb ("do NOT wobble", "WITHOUT flailing", "AVOID bouncing")
    # describes what to AVOID while holding still — it is NOT the active objective, so it must
    # not veto a balance goal's static_hold (that false-rejected honest one-leg/still metrics).
    _NEG = {"not", "dont", "don", "no", "without", "never", "avoid", "stop", "cease",
            "minimize", "minimise", "reduce", "resist", "prevent"}
    for i, t in enumerate(seq):
        stem = t.rstrip("s")
        ing = t[:-3] if t.endswith("ing") else None
        ed = t[:-2] if t.endswith("ed") else None
        is_active = (t in active_set or t in directional or stem in active_stems
                     or (ing is not None and ing in active_stems)
                     or (ed is not None and ed in active_stems))
        if not is_active:
            continue
        if any(w in _NEG for w in seq[max(0, i - 3):i]):
            continue   # negated motion → AVOID instruction, not the objective
        return True
    return False


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


def _spec_has_commanded_motion(spec: Any, *, dynamic_only: bool = False) -> bool:
    """True iff the spec commands joint/whole-body MOTION — a pelvis fold, a tremor/noise
    whole-body channel, travel/hops, OR any group with a real amplitude/peak/burst (an
    oscillate/burst). With `dynamic_only=False` (default) a held distinctive posture (a 'hold'
    group offset, e.g. a raised arm) ALSO counts — a salute hold is NOT do_nothing, so
    static-hold must treat it as active and KEEP do_nothing. With `dynamic_only=True` a
    standalone hold offset does NOT count (it renders with ZERO joint velocity → a settled
    static posture, not motion) — terminal-down uses this so a lie-down with a settled limb
    (or an active duck holding a bent-leg posture) is not mis-flipped to active by a
    zero-velocity offset (§round-21: that false-rejected an honest descend-and-rest metric).
    Posture alone (uprightness + base_height) is NEVER motion. §round-18 thresholds."""
    try:
        if abs(float(getattr(spec, "fold_depth_m", 0.0) or 0.0)) > 0.05:
            return True
        if abs(float(getattr(spec, "tremor", 0.0) or 0.0)) > 0.1:
            return True   # tremor is a whole-body motion channel (NOT per-group)
        if abs(float(getattr(spec, "noise", 0.0) or 0.0)) > 0.02:
            return True   # noise (gaussian on joint_vel) is whole-body motion too
        if abs(float(getattr(spec, "forward_speed_mps", 0.0) or 0.0)) > 0.1:
            return True
        if abs(float(getattr(spec, "lateral_speed_mps", 0.0) or 0.0)) > 0.1:
            return True
        if int(getattr(spec, "hop_count", 0) or 0) > 0 and float(getattr(spec, "hop_height_m", 0.0) or 0.0) > 0.05:
            return True
        for gr in (getattr(spec, "groups", []) or []):
            amp = abs(float(getattr(gr, "amplitude_rad", 0.0) or 0.0))
            peak = abs(float(getattr(gr, "peak_radps", 0.0) or 0.0))
            burst = int(getattr(gr, "burst_count", 0) or 0)
            off = abs(float(getattr(gr, "offset_rad", 0.0) or 0.0))
            if amp > 1e-3 or peak > 1e-3 or burst > 0:
                return True
            if off > 1e-2 and not dynamic_only:   # a held offset = a distinctive POSTURE
                return True
        return False
    except Exception:  # noqa: BLE001 — unparseable spec → assume motion (the safe direction:
        return True     # keep the velocity/posture defense)


def _spec_is_static_hold(spec: Any) -> bool:
    """True iff a competence-ladder TOP rung describes a STILL UPRIGHT hold — high
    uprightness, NOMINAL standing height (base_height_m≈0.7, no ramp), and no commanded joint
    motion. Reads the blind author's own notion of competence (anti-collusion: the author
    never sees the metric), so it decides whether do_nothing_upright is ON-goal WITHOUT
    brittle goal-text keywords.

    §round-19: base_height_m is read too — a base_height RAMP is commanded VERTICAL MOTION and
    a held NON-nominal height (squat) is a DIFFERENT posture where standing-still is OFF-goal.
    A genuine balance metric (scores do_nothing HIGH) authors a NOMINAL-height hold (≥0.55, no
    ramp) → still static; a ramp/squat metric scores do_nothing LOW → keeping the loser cannot
    deny it. (NOTE: a crouch→stand TRANSITION whose TOP rung is a held standing posture also
    passes this PER-RUNG test — that is suppressed at the LADDER level via _ladder_has_crouched_rung.)"""
    try:
        if _scalar_end(getattr(spec, "uprightness", 1.0)) < 0.8:
            return False
        bh = getattr(spec, "base_height_m", 0.7)
        if _scalar_start(bh) < 0.55 or _scalar_end(bh) < 0.55:
            return False
        if abs(_scalar_end(bh) - _scalar_start(bh)) > 0.05:
            return False
        return not _spec_has_commanded_motion(spec)
    except Exception:  # noqa: BLE001 — unparseable spec → not static-hold (keep losers)
        return False


def _ladder_has_crouched_rung(rungs: Any) -> bool:
    """§round-20: True iff any authored rung is a held LOW-but-UPRIGHT posture (a crouch/squat
    TARGET). A crouch/sit→stand TRANSITION ladder has a held-standing TOP rung (which passes the
    per-rung static-hold test) but low/crouched LOWER rungs — its goal is to RISE, so do_nothing
    (already standing, never rose) is OFF-goal and must be KEPT.

    §round-21: a low rung is only a crouch TARGET when it is ALSO UPRIGHT (torso vertical). A
    blind balance author naturally renders a FALL/stumble failure rung as a pelvis DROP — low
    base_height AND low uprightness — which is NOT a crouch target, just the bottom of a balance
    ladder. Counting it suppressed static_hold and false-rejected an honest balance metric, so
    require uprightness ≥ 0.7 before treating a low rung as a crouch."""
    try:
        for r in (rungs or []):
            bh = getattr(r, "base_height_m", 0.7)
            if (min(_scalar_start(bh), _scalar_end(bh)) < 0.55
                    and _scalar_end(getattr(r, "uprightness", 1.0)) >= 0.7):
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _spec_is_terminal_down(spec: Any) -> bool:
    """True iff a competence-ladder TOP rung describes a DOWN end-state (lie/rest) —
    so collapse_and_stay_down is ON-goal and is dropped from the loser set.

    §round-19 [HIGH FALSE GRANT] fix (sibling asymmetry to _spec_is_static_hold's f2932eb
    base_height fix): a DOWN end-state is NON-UPRIGHT. A near-floor height ALONE is NOT
    terminal-down — a held UPRIGHT deep squat (low pelvis, torso vertical) is an ACTIVE low
    posture, NOT a lie/rest, and must KEEP collapse_and_stay_down (the only loser that catches
    a dip-DEPTH-only proxy gamed by collapsing). The old unguarded `base_height_m <= 0.35`
    OR-branch fired regardless of uprightness, so a squat-and-hold goal dropped collapse and a
    depth-only confound false-granted."""
    try:
        up = _scalar_end(getattr(spec, "uprightness", 1.0))
        bh = _scalar_end(getattr(spec, "base_height_m", 0.7))
        # clearly non-upright (lying/fallen, any height) OR near-floor AND not-upright
        # (a low collapsed heap). An upright squat (up high, bh low) is NEITHER.
        down = (up <= 0.3) or (bh <= 0.35 and up <= 0.5)
        # §round-20: a genuine lie/REST end-state is STILL. A low posture WITH DYNAMIC motion
        # (writhe/thrash/roll/worm — an oscillate/burst/tremor) is an ACTIVE low goal, NOT
        # lie/rest — collapse_and_stay_down must stay (an honest moving-low metric scores it ~0)
        # and collapse_and_thrash must NOT be injected (it IS the on-goal end-state of a writhe).
        # §round-21: dynamic_only=True so a settled limb HOLD-offset (zero joint velocity) does
        # NOT flip a genuine lie-down to active (that false-rejected an honest descend-and-rest);
        # the descent-magnitude channel is still probed on the terminal-down path because
        # collapse_and_thrash now RAMPS the pelvis from nominal to the floor.
        return down and not _spec_has_commanded_motion(spec, dynamic_only=True)
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
    travels: Optional[bool] = None, hops: Optional[bool] = None,
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

    def _pack(name: str, channel: str, spec: Any, idx: int,
              max_frac: Optional[float] = None, reference_only: bool = False) -> dict:
        arrays, behavior, meta = render_rung(spec, joint_names, rung_index=idx)
        d = {"name": name, "channel": channel, "arrays": arrays,
             "behavior": behavior, "meta": dict(meta)}
        if max_frac is not None:
            d["max_frac"] = max_frac
        if reference_only:
            d["reference_only"] = True   # scored for a paired check, NOT ceiling-compared
        return d

    sh = _goal_is_static_hold(behavior_goal) if static_hold is None else static_hold
    td = _goal_is_terminal_down(behavior_goal) if terminal_down is None else terminal_down
    tr = _goal_is_locomotion(behavior_goal) if travels is None else travels
    hp = _goal_is_jump(behavior_goal) if hops is None else hops
    losers: list[dict] = []
    if not (sh and not td):   # still-upright losers are on-goal ONLY for a balance goal
        # do_nothing is PERFECTLY still + upright and performs NO active goal, so an honest
        # metric of any active goal MUST score it ~0. §round-21: it carries a strict per-loser
        # FLOOR (max_frac) — a metric that pays this pure-idle anchor even a fraction of
        # competence (an additive uprightness-gated floor: `up·(FLOOR + (1-FLOOR)·goal)`) is
        # GAMEABLE, because the gate's upper ceiling (≈0.5) otherwise left an open [0,0.5)
        # band a do-nothing policy could farm. An honest multiplicative/min metric (LAW: the
        # objective metric is a completion-gate·min(channels), not an additive sum) scores
        # do_nothing 0 and clears the floor.
        losers.append(_pack("do_nothing_upright", "posture",
                            MotionSpec(uprightness=1.0, base_height_m=0.7), 600,
                            max_frac=_LOSER_FLOOR_FRAC))
        # a SMALL twitch (tremor 0.04 ≈ 0.07 rad ROM, mean|jv| ~1.18, peak ~3.5 rad/s): enough
        # joint VELOCITY at 15 Hz to trip a "rewards any motion" proxy, ROM far below any
        # plausible gesture target so an honest small-amplitude gesture metric is not denied.
        # NO flat floor (unlike do_nothing): an active velocity goal legitimately scores jitter
        # moderately. §round-26: the additive raw-MEAN-velocity floor confound that farms ~46%
        # via this probe under the 0.5 ceiling is now caught by the PEAK discriminator below
        # (velocity_peak_ref) — a flat floor was NOT viable (honest GOOD_KICK pays jitter 0.52 of
        # competence, ABOVE the confound's 0.48, so the means don't separate), but jitter's PEAK
        # (~3.5) vs a competent-intensity motion's peak (~15–20) DOES.
        losers.append(_pack("jitter_in_place", "posture",
                            MotionSpec(uprightness=1.0, base_height_m=0.7,
                                       tremor=0.04), 601))
        # §round-26 PEAK DISCRIMINATOR (reference-only — never ceiling-compared): a WHOLE-BODY
        # tremor at 2.5× jitter's amplitude — SAME joint coverage as jitter (so a mean-ROM-over-
        # all-joints honest metric treats them comparably) but ~2.5× the PEAK velocity (~8.7 vs
        # jitter's ~3.5) and ~2.5× the ROM. It pairs with jitter_in_place: an additive MEAN-
        # velocity-floor confound's `1−exp(−mean/k)` SATURATES, so it scores jitter ≈
        # velocity_peak_ref — INSENSITIVE to the 2.5× peak/ROM difference (the farming signature).
        # A genuine velocity/kick metric (peak-sensitive) OR a ROM/amplitude metric scores
        # velocity_peak_ref HIGHER than jitter. The ratio is INVARIANT to the confound's FLOOR_v
        # (both probes scale together), so the discriminator is robust to tuning, unlike a flat
        # floor (which can't separate the confound's 0.46 from honest GOOD_KICK's 0.52 jitter
        # score). See the gate's velocity-floor verdict.
        losers.append(_pack("velocity_peak_ref", "velocity",
                            MotionSpec(uprightness=1.0, base_height_m=0.7, tremor=0.10),
                            610, reference_only=True))
        # §round-33 [HIGH FALSE GRANT] fix: a SECOND idle-posture FLOOR probe at a point ADJACENT to
        # do_nothing's EXACT signature — a near-idle upright stance with a tiny low-frequency tremor
        # (meanjv > 0, ROM ~tiny). do_nothing_upright is perfectly still (meanjv==0), so a confound
        # can SURGICALLY raise on its exact signature ("if meanjv==0.0 and up>=0.999: raise") to skip
        # the do_nothing FLOOR while ROM-gating a flat posture credit below jitter's ROM (defeating
        # the round-32 do_nothing_farm jitter≥ref refinement, which reasons over the velocity ladder
        # — orthogonal to a joint-POSITION-ROM farm). near_still is OFF that exact point (meanjv>0),
        # so the exact-signature raise does NOT fire here → the floor catches the farm; a band-keyed
        # raise wide enough to cover near_still also raises at runtime on the real near-idle policies
        # it would farm (self-defeating). An honest onset/timing metric has no event to time here →
        # raises → counted 0.0 → floor passes (round-15); an honest ACTIVE metric reads the goal's
        # joints (tiny here) → ~0. Carries the same per-loser FLOOR as do_nothing.
        losers.append(_pack("near_still_upright", "posture",
                            MotionSpec(uprightness=1.0, base_height_m=0.7, tremor=0.001), 606,
                            max_frac=_LOSER_FLOOR_FRAC))
    # §round-32 [CRITICAL FALSE GRANT] fix: an upright policy that TRAVELS forward while
    # performing NONE of the in-place goal — the off-goal horizontal-travel channel that every
    # prior loser leaves wholly UNCOVERED (do_nothing/jitter/ref are stationary-upright; collapse
    # is toppled-in-place; all four have 0 m base travel and all stationary ladder rungs do too).
    # So an additive SUM `up·(α·goal + β·travel)` ranks the (non-traveling) ladder via the goal
    # term while the dormant β·travel term invisibly farms a run-forward policy (scored ≥ the
    # ceiling). An honest metric of an active STATIONARY goal reads the goal's joints/posture —
    # walk_away has zero joint motion and nominal height/uprightness, so it scores ~0. KEPT only
    # for an ACTIVE, STATIONARY, NON-balance goal: dropped for a locomotion goal (travel is
    # on-goal), a balance goal (an honest uprightness-only balance metric scores an upright
    # traveler high → would be false-rejected), and a lie goal (an honest low metric scores an
    # upright traveler ~0 anyway). The DURABLE close of the additive-SUM class is the
    # min-composition law; this probe closes the specific (and large) locomotion channel.
    if not tr and not sh and not td:
        # travels DIAGONALLY (forward AND lateral) so a farm keyed on root x-range, y-range, OR the
        # xy-norm all read a large displacement → caught regardless of the travel axis; an in-place
        # honest metric reads the goal's joints (zero here) and nominal height/uprightness → ~0.
        losers.append(_pack("walk_away_upright", "locomotion",
                            MotionSpec(uprightness=1.0, base_height_m=0.7,
                                       forward_speed_mps=1.5, lateral_speed_mps=1.0), 605))
    # §round-33 [HIGH FALSE GRANT] fix: an upright policy that HOPS VERTICALLY in place — the off-goal
    # VERTICAL root-z channel walk_away leaves uncovered (it travels horizontally with root z held
    # flat). An additive SUM `up·(α·goal + β·hop)` keyed on root-z bounce amplitude ranks the
    # (flat-z) ladder via the goal term while the dormant β·hop term farms a vertical bobber that
    # performs none of the goal (scored ≥ ceiling). An honest in-place metric reads the goal's joints
    # (zero here) → ~0. KEPT for an ACTIVE, STATIONARY, NON-balance, NON-jump goal; dropped for a
    # JUMP/hop goal (vertical hop is on-goal — derived from the blind ladder's commanded hops via
    # _ladder_hops, keyword fallback _goal_is_jump), a balance goal (an uprightness-only metric scores
    # an upright hopper high), and a lie goal.
    if not hp and not sh and not td:
        losers.append(_pack("hop_in_place_upright", "vertical",
                            MotionSpec(uprightness=1.0, base_height_m=0.7,
                                       hop_height_m=0.5, hop_count=4), 607))
    if not td:
        losers.append(_pack("collapse_and_stay_down", "completion",
                            MotionSpec(uprightness=0.0, base_height_m=[0.7, 0.1]), 602))
    else:
        # §round-19 [FALSE GRANT] fix: for a terminal-DOWN goal, collapse_and_stay_down is
        # on-goal (dropped — an honest lie-down/still-crouch metric LEGITIMATELY scores a
        # still low policy high), and do_nothing/jitter are rendered UPRIGHT (z=0.7), which an
        # honest low-posture metric scores ~0. So NOTHING probes the low posture itself. TWO
        # thrashing probes cover it without false-rejecting an honest STILL-low metric (both
        # THRASH → an honest low·stillness metric scores them ~0):
        #   • collapse_and_thrash — a CONSTANT-low pelvis + thrash → catches a low-height-ONLY
        #     proxy (it scores the held-low max) that ignores 'rest still'.
        #   • descend_and_thrash — a pelvis RAMP nominal→floor + thrash → catches a
        #     descent-MAGNITUDE confound (start−min drop) on a controlled-crouch/duck goal that
        #     a constant-low probe scores 0. (§round-21: a single ramped probe could not be
        #     both maximally-low AND maximally-descending, so the two channels need two probes.)
        losers.append(_pack("collapse_and_thrash", "stillness",
                            MotionSpec(uprightness=0.0, base_height_m=0.12,
                                       tremor=1.8, noise=0.15), 603))
        losers.append(_pack("descend_and_thrash", "completion",
                            MotionSpec(uprightness=0.0, base_height_m=[0.7, 0.12],
                                       tremor=1.8, noise=0.15), 604))
    return losers


# ── §round-35 FIX A: goal-channel SCOPING via off-goal-perturbation invariance ──
#
# The per-channel required-losers (do_nothing/jitter/walk_away/hop/near_still/collapse) close ONE
# off-goal channel each, and rounds 32-34 proved that is unbounded whack-a-mole — every round found a
# fresh uncovered channel (travel → hop → dip → WHOLE-BODY-ROM) a confound farms via an additive SUM
# OR a MULTIPLICATIVE gate·channel (the latter defeats the min-composition law alone). The DURABLE
# fix is goal-channel SCOPING: a genuinely goal-scoped metric's score must NOT RISE when the OFF-goal
# channels are perturbed on a COMPETENT reference. Off-goal = every joint NOT in the blind ladder's
# groups' role_query, plus every root DOF the ladder does not command (travel/hop/dip). This closes
# ALL uncovered channels at once (joints AND root DOF, additive AND multiplicative) and makes the
# posture/terminal-down/idle carve-outs SAFE (a collapse/idle/flail confound reads off-goal channels
# → caught regardless of the keyword classifier). Designed + empirically validated (a 3-variant design
# workflow): 4/4 known farms CAUGHT (min rise +0.536), 9/9 honest corpus metrics PASS (max rise
# +0.000). NEVER raises, NEVER denies on absence of evidence (an error / no usable competence is
# inconclusive). uprightness (projected_gravity_b) is the one channel NEVER perturbed (on-goal posture
# for every upright goal — perturbing it would false-reject every upright honest metric), so a metric
# reading ONLY uprightness is invariant (rise 0) and PASSES — it is the documented shared-channel case.
_PERTURB_MARGIN = 0.25     # deny if off-goal motion RAISES the score by ≥ this (honest rise is ~0.0)
_PERTURB_REL_FRAC = 0.5    # ...OR claims ≥ this fraction of the remaining headroom (1−comp): catches
_PERTURB_ABS_FLOOR = 0.03  #    a ceiling-SATURATED gate whose absolute rise is bounded by 1−comp,
#                              gated by a small absolute floor so an honest saturated metric (rise 0) is safe
_PERTURB_DT = 0.02
#: §round-37 (round-36 defect 3): for a VELOCITY-characterized goal, a metric that RETAINS ≥ this
#: fraction of its competent score under a goal-joint SLOW-DOWN (same ROM, ~zero velocity) reads only
#: ROM/position, not the goal's required burst speed → gameable by a slow large-ROM sweep. GOOD_KICK
#: retains ~0.26; a pure-ROM confound retains ~1.0 — so 0.6 separates cleanly with a wide margin.
#: RESIDUAL: velocity-characterization is read from the ladder groups' burst mode (peak_radps); a
#: SPEED goal MIS-RENDERED as a high-frequency amplitude-graded `oscillate` ladder (peak_radps=0)
#: escapes the slow-down check (round-38 — a mismatched-ladder residual: the ladder grades amplitude,
#: so a ROM metric is "calibrated" to it; rendered peak velocity can't separate it from an honest
#: amplitude wave, which is also fast). LOW threat (the blind author would render a kick as burst).
_PERTURB_SLOW_RETAIN = 0.6
#: §round-39…44: GOAL-JOINT SENSITIVITY (the MIN-COMPOSITION LAW) — a granted metric must read its
#: DECLARED goal joints, not a root/posture proxy. An honest metric is `completion_gate·min/product
#: (channels)`, so stilling the goal joints collapses the score; a farm `α·goal + (1−α)·root` retains
#: ~(1−α). The verdict is CEILING-AWARE (§round-44): gameable iff the goal-joint-STILLED score `gneut`
#: ≥ the gaming ceiling `min(_ADV_REL_CEIL·comp, _ADV_ABS_CEIL)` — i.e. "doing the on-goal root but
#: NOT the goal joints still looks competent." This supersedes the round-41/42 fixed drop bar (0.4→0.8),
#: which was a knife-edge (0.4 fell to a 0.4·goal+0.6·root split) AND double-counted an ON-goal root
#: channel (false-rejecting an honest `0.7·legROM + 0.3·pelvis-descent` squat whose stilled residual
#: 0.30 is sub-ceiling — round-43). A useful root farm (residual ≥ ceiling) is flagged; an honest
#: metric whose non-goal residual is sub-ceiling passes. Skipped for a pure-posture goal (no goal joints).


def _derive_goal_channels(valid_ladders: Any, names: list[str]) -> tuple[set, set]:
    """GOAL JOINTS = joints the blind ladder groups' role_query targets (the same select_joints
    resolver render_rung uses — anti-collusion-safe, never sees the metric). ON-GOAL ROOT DOF = which
    root channels the ladder commands (travel x/y, hop z-up, fold/crouch z-down). root-z is ONE
    physical DOF: if vertical motion is on-goal in EITHER direction it is wholly on-goal (else an
    honest hop metric reading the root-z RANGE would read the off-goal downward dip)."""
    from sculptor.eval.joint_resolver import select_joints
    rung_lists = [list(getattr(L, "rungs", L) or []) for L in valid_ladders]
    goal_joints: set[int] = set()
    for rungs in rung_lists:
        for rung in rungs:
            for gr in (getattr(rung, "groups", []) or []):
                rq = getattr(gr, "role_query", None)
                if rq is None:
                    continue
                # §round-39: only ACTIVE-motion groups define goal joints. A `hold` group (a settled
                # static posture offset — amplitude_rad=peak_radps=0, round-21's settled-limb case) is
                # POSTURE, not a motion the metric must read; counting it would false-reject an honest
                # lie/rest metric (reads height+stillness, not the held limb) via the goal-joint
                # sensitivity check.
                if (abs(float(getattr(gr, "amplitude_rad", 0.0) or 0.0)) <= 1e-9
                        and abs(float(getattr(gr, "peak_radps", 0.0) or 0.0)) <= 1e-9):
                    continue
                goal_joints.update(select_joints(
                    names, segments=(rq.segments or None),
                    axes=(list(rq.axes) if rq.axes else None),
                    sides=(rq.sides or None)))
    on_root: set[str] = set()
    # §round-37 [MEDIUM FALSE GRANT fix, round-36 defect 2]: travel (x/y) is on-goal ONLY for a
    # GENUINE single-axis locomotion ladder — one with NO joint competence (goal_joints empty). A
    # blind author co-varying a JOINT goal (e.g. a left kick) with an incidental forward step makes
    # _ladder_travels True; trusting that unconditionally marked travel on-goal → scope SKIPPED the
    # travel perturbation → a pure-travel confound escaped. Gating on `not goal_joints` keeps honest
    # locomotion (no groups → travel on-goal → invariant → grants) while perturbing the incidental
    # travel of a joint-competence ladder (→ a travel farm rises → caught). hop/dip stay ladder-
    # derived: a fold/squat ladder genuinely has joint groups AND an on-goal pelvis dip, so gating
    # zdn on `not goal_joints` would false-reject honest fold metrics.
    if _ladder_travels(rung_lists) and not goal_joints:
        on_root.update({"x", "y"})
    if _ladder_hops(rung_lists):
        on_root.add("zup")
    all_rungs = [r for rungs in rung_lists for r in rungs]
    has_fold = any(abs(float(getattr(r, "fold_depth_m", 0.0) or 0.0)) > 0.05 for r in all_rungs)
    if has_fold or _ladder_has_crouched_rung(all_rungs):
        on_root.add("zdn")
    if "zup" in on_root or "zdn" in on_root:
        on_root.update({"zup", "zdn"})
    return goal_joints, on_root


def _perturb_off_goal(arrays: dict, goal_joints: set, on_root: set, *,
                      amp: float = 1.2, period: int = 18, travel: float = 1.5,
                      hop: float = 0.5, dip: float = 0.4) -> dict:
    """Copy the competent arrays; leave GOAL joints + projected_gravity_b (uprightness) + the ON-goal
    root DOF UNCHANGED; ADD large motion to every OFF-goal joint and every OFF-goal root DOF."""
    jp = np.asarray(arrays["joint_pos"]).copy()
    jv = np.asarray(arrays["joint_vel"]).copy()
    root = np.asarray(arrays["root_link_pos_w"]).copy()
    T, _, J = jp.shape
    t = np.arange(T)
    off_joints = [j for j in range(J) if j not in goal_joints]
    # §round-37 [round-36 defect 1, partial]: a MULTI-BAND off-goal drive (a sum of several periods)
    # so an off-goal reader keyed to a single common frequency band is excited, not just a period-18
    # one. A razor-sharp matched filter at an OFF-GRID period on an off-goal joint remains a documented
    # LOW-threat residual — the metric source is SYSTEM-generated (it does not contain pathological
    # narrow-band matched filters on off-goal joints), and chasing every frequency is unbounded; the
    # durable spirit is goal-channel min-composition, not frequency whack-a-mole. ROM/velocity/whole-
    # body-ROM readers are caught by the large multi-band amplitude regardless of band.
    _BANDS = (18, 7, 11, 29, 43)
    for n, j in enumerate(off_joints):
        sig = np.zeros(T)
        for bi, per in enumerate(_BANDS):
            sig += amp * np.sin(2 * np.pi * t / per + 0.3 * n + 0.5 * bi)
        jp[:, :, j] += sig[:, None]
    jv2 = np.gradient(jp, axis=0) / _PERTURB_DT
    for j in off_joints:
        jv[:, :, j] = jv2[:, :, j]
    if "x" not in on_root:
        root[..., 0] = root[..., 0] + np.cumsum(np.full(T, travel * _PERTURB_DT))[:, None]
    if "y" not in on_root:
        root[..., 1] = root[..., 1] + np.cumsum(np.full(T, travel * _PERTURB_DT))[:, None]
    z = root[:, 0, 2].copy()
    if "zup" not in on_root:
        for c in range(4):
            s0 = c * 30 + 5
            for k in range(20):
                if s0 + k < T:
                    z[s0 + k] += hop * np.sin(np.pi * k / 20)
    if "zdn" not in on_root:
        arc = (1.0 - np.cos(2.0 * np.pi * t / T)) / 2.0
        z = np.clip(z - dip * arc, 0.0, None)
    root[..., 2] = z[:, None]
    out = dict(arrays)
    out["joint_pos"], out["joint_vel"], out["root_link_pos_w"] = jp, jv, root
    return out


def _ladder_is_velocity_mode(valid_ladders: Any) -> bool:
    """§round-37 (round-36 defect 3): True iff the goal is VELOCITY-characterized — the blind ladder
    grades competence by burst SPEED (any group with peak_radps > 0, i.e. mode='burst'). For these a
    fast burst IS the goal; a SLOW large-ROM goal-joint sweep is degenerate. For ROM/amplitude-
    characterized goals (fold/oscillate, peak_radps=0) a slow large motion is on-goal, so the slow-down
    check is NOT applied (it would false-reject honest fold/ROM metrics)."""
    for L in (valid_ladders or []):
        for rung in (getattr(L, "rungs", L) or []):
            for gr in (getattr(rung, "groups", []) or []):
                if abs(float(getattr(gr, "peak_radps", 0.0) or 0.0)) > 1e-6:
                    return True
    return False


def _slow_goal_joints(arrays: dict, goal_joints: set) -> dict:
    """§round-37: replace the GOAL joints' trajectories with a SLOW ramp through their per-env
    [min, max] range — same ROM/amplitude, ~zero velocity. A velocity/burst metric DROPS (its peak
    collapses); a ROM-only metric is INVARIANT (it reads the unchanged range). Off-goal channels +
    uprightness are left as the competent reference."""
    jp = np.asarray(arrays["joint_pos"]).copy()
    jv = np.asarray(arrays["joint_vel"]).copy()
    T = jp.shape[0]
    ramp = np.linspace(0.0, 1.0, T)
    for j in goal_joints:
        lo = jp[:, :, j].min(axis=0)
        hi = jp[:, :, j].max(axis=0)
        jp[:, :, j] = lo[None, :] + ramp[:, None] * (hi - lo)[None, :]
    jv2 = np.gradient(jp, axis=0) / _PERTURB_DT
    for j in goal_joints:
        jv[:, :, j] = jv2[:, :, j]
    out = dict(arrays)
    out["joint_pos"], out["joint_vel"] = jp, jv
    return out


def _neutralize_goal_joints(arrays: dict, goal_joints: set) -> dict:
    """§round-39 (round-38 defects 2+3): STILL the GOAL joints (hold them at frame 0, zero velocity),
    leaving everything else (off-goal joints, root, uprightness) as the competent reference. If the
    metric's score does NOT drop, it is INSENSITIVE to its declared goal joints — it reads a root/
    posture channel instead (a pelvis-bob/dip farm) → gameable. An honest goal-scoped metric (reads
    its goal joints' ROM or velocity) drops to ~0."""
    jp = np.asarray(arrays["joint_pos"]).copy()
    jv = np.asarray(arrays["joint_vel"]).copy()
    for j in goal_joints:
        jp[:, :, j] = jp[0:1, :, j]
        jv[:, :, j] = 0.0
    out = dict(arrays)
    out["joint_pos"], out["joint_vel"] = jp, jv
    return out


def _neutralize_off_goal(arrays: dict, goal_joints: set, on_root: set) -> dict:
    """§round-37 (round-36 defect 2): the REMOVE-direction companion of _perturb_off_goal. REMOVE the
    competent reference's OFF-goal motion (still the off-goal joints at their first frame; freeze the
    off-goal root DOF at their start value), leaving GOAL joints + uprightness + ON-goal root
    unchanged. If the metric's score DROPS when off-goal motion is removed, the score was ELEVATED BY
    an off-goal channel the competent reference happens to exercise (e.g. an incidental forward step
    in a mis-rendered kick ladder) → the metric REWARDS off-goal motion → gameable. An honest metric
    that PENALIZES off-goal (e.g. a stationarity gate) scores HIGHER when off-goal is removed (no
    drop) → not flagged; a goal-scoped metric is invariant."""
    jp = np.asarray(arrays["joint_pos"]).copy()
    jv = np.asarray(arrays["joint_vel"]).copy()
    root = np.asarray(arrays["root_link_pos_w"]).copy()
    _, _, J = jp.shape
    off_joints = [j for j in range(J) if j not in goal_joints]
    for j in off_joints:
        jp[:, :, j] = jp[0:1, :, j]   # hold first frame → still
        jv[:, :, j] = 0.0
    if "x" not in on_root:
        root[..., 0] = root[0:1, ..., 0]
    if "y" not in on_root:
        root[..., 1] = root[0:1, ..., 1]
    if "zup" not in on_root and "zdn" not in on_root:
        root[..., 2] = root[0:1, ..., 2]   # freeze root height (remove off-goal hop/dip)
    out = dict(arrays)
    out["joint_pos"], out["joint_vel"], out["root_link_pos_w"] = jp, jv, root
    return out


def off_goal_perturbation_verdict(
    gen_fn: Any, roles: Any, valid_ladders: Any, names: list[str],
    *, margin: float = _PERTURB_MARGIN,
) -> dict[str, Any]:
    """§Fix A: score the metric on a COMPETENT reference vs (a) an OFF-GOAL-PERTURBED copy (off-goal
    motion ADDED) and (b) an OFF-GOAL-NEUTRALIZED copy (off-goal motion REMOVED). A goal-scoped metric
    is INVARIANT to both; a confound that REWARDS an off-goal channel either RISES when it is added or
    DROPS when it is removed (gameable iff max(rise, drop) clears the margin). NEVER raises; on any
    failure / no usable competence returns gameable=False (inconclusive — never a deny)."""
    rec: dict[str, Any] = {"ran": False, "gameable": False, "comp": None, "pert": None}
    try:
        from sculptor.eval.ladder_synth import render_ladder
        rungs = list(getattr(valid_ladders[0], "rungs", valid_ladders[0]) or [])
        synth = render_ladder(rungs, names)
        if synth.get("degenerate") or not synth.get("rungs"):
            return rec
        arrays, behavior, meta = synth["rungs"][-1]    # top (competent) rung
        meta = dict(meta); inject_joint_roles(meta, roles)
        comp = _gameable_score(float(gen_fn(arrays, behavior, meta).get("spec_score", 0.0)))
        if not np.isfinite(comp) or comp <= 0.0:
            return rec                                  # no usable competence anchor → inconclusive
        goal_joints, on_root = _derive_goal_channels(valid_ladders, names)
        if not goal_joints and not on_root:
            # a pure-POSTURE goal (balance/lie — no joint/root channel): uprightness is unperturbed,
            # so an honest posture metric is invariant; still run (a whole-body-ROM confound rises).
            pass
        parr = _perturb_off_goal(arrays, goal_joints, on_root)
        pmeta = dict(meta); inject_joint_roles(pmeta, roles)
        pert = _gameable_score(float(gen_fn(parr, behavior, pmeta).get("spec_score", 0.0)))
        # §round-37: the REMOVE direction (round-36 defect 2) — if the score DROPS when the off-goal
        # motion the competent reference exercises is removed, the score was elevated by an off-goal
        # channel → the metric rewards off-goal → gameable. Caught even when ADDING off-goal saturates.
        narr = _neutralize_off_goal(arrays, goal_joints, on_root)
        nmeta = dict(meta); inject_joint_roles(nmeta, roles)
        neut = _gameable_score(float(gen_fn(narr, behavior, nmeta).get("spec_score", 0.0)))
        # §round-37 (round-36 defect 3): for a VELOCITY-characterized goal, also SLOW the goal joints
        # (same ROM, ~zero velocity). A burst/velocity metric drops; a ROM-only metric is invariant →
        # it reads only the on-goal channel's RANGE, not the goal's required SPEED → gameable by a slow
        # large-ROM sweep. Only for burst goals + with goal joints (a fold/ROM goal's slow motion is
        # on-goal, so this is NOT applied — it would false-reject honest fold metrics).
        slow_retained = None
        if goal_joints and _ladder_is_velocity_mode(valid_ladders):
            sarr = _slow_goal_joints(arrays, goal_joints)
            smeta = dict(meta); inject_joint_roles(smeta, roles)
            slow = _gameable_score(float(gen_fn(sarr, behavior, smeta).get("spec_score", 0.0)))
            slow_retained = slow / comp if comp > 0 else 0.0
        # §round-39 (round-38 defects 2+3) / §round-44 (round-43 false-reject fix): GOAL-JOINT
        # SENSITIVITY — still the goal joints (keep the ON-goal root + uprightness); the metric is
        # gameable iff a goal-joint-stilled policy STILL scores in COMPETENT territory (gneut ≥ the
        # gaming ceiling). This is CEILING-AWARE (replaces the round-42 fixed 0.8·comp drop bar, which
        # double-counted an ON-goal root channel and false-rejected an honest additive squat
        # `0.7·legROM + 0.3·pelvis-descent`: stilling the legs left the on-goal descent 0.30 — below
        # the ceiling, NOT a useful farm). A token-split root farm leaves (1−α)≥0.5 ≥ ceiling → flagged;
        # an honest metric whose on-goal-root residual is sub-ceiling passes. Semantics: "doing the
        # on-goal root but NOT the goal joints must not look competent."
        goal_drop = None
        gneut = None
        if goal_joints:
            garr = _neutralize_goal_joints(arrays, goal_joints)
            gmeta = dict(meta); inject_joint_roles(gmeta, roles)
            gneut = _gameable_score(float(gen_fn(garr, behavior, gmeta).get("spec_score", 0.0)))
            goal_drop = comp - gneut
    except Exception:  # noqa: BLE001 — never raises, never denies on absence of evidence
        return rec
    rise = pert - comp           # ADD off-goal → score rises (rewards added off-goal motion)
    drop = comp - neut           # REMOVE off-goal → score drops (relied on off-goal motion)
    headroom = max(1.0 - comp, 0.0)
    swing = max(rise, drop)
    gameable = (swing >= margin) or (swing >= _PERTURB_REL_FRAC * headroom and swing > _PERTURB_ABS_FLOOR)
    on_goal_char = bool(slow_retained is not None and slow_retained >= _PERTURB_SLOW_RETAIN)
    ceiling = min(_ADV_REL_CEIL * comp, _ADV_ABS_CEIL)
    goal_joint_insensitive = bool(gneut is not None and gneut >= ceiling)
    gameable = gameable or on_goal_char or goal_joint_insensitive
    rec.update(ran=True, gameable=bool(gameable), comp=round(comp, 4), pert=round(pert, 4),
               neut=round(neut, 4), rise=round(rise, 4), drop=round(drop, 4),
               slow_retained=(round(slow_retained, 4) if slow_retained is not None else None),
               on_goal_char=on_goal_char,
               goal_drop=(round(goal_drop, 4) if goal_drop is not None else None),
               gneut=(round(gneut, 4) if gneut is not None else None),
               ceiling=round(ceiling, 4), goal_joint_insensitive=goal_joint_insensitive,
               n_goal_joints=len(goal_joints), on_root=sorted(on_root))
    return rec


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
    # §round-21: a per-loser FLOOR breach (a pure-idle probe scoring ≥ max_frac·competent_ref).
    floor_gamed, floor_name, floor_thresh = False, None, None
    # §round-26: scored deterministic losers + paired reference probes, for the velocity-floor check.
    loser_scores: dict[str, float] = {}
    ref_scores: dict[str, float] = {}
    # §round-31 (UNIFIED probe-raise fail-open closure): the verdict-feeding probes — do_nothing_upright
    # (per-loser FLOOR), jitter_in_place (velocity-floor numerator), velocity_peak_ref (velocity-floor
    # denominator). A SURGICAL raise on one of these — keyed on its recognizable single-point signature
    # (do_nothing: up=1+still; jitter: mean|jv|≈1.18; ref: mean|jv|≈2.94) so real rungs are unaffected —
    # used to leave the probe unscored → its verdict SILENTLY SKIPPED (fail-OPEN: round-30 D1 = ref,
    # round-31 A1 = jitter, round-31 = do_nothing floor). An HONEST onset/phase metric that can't time a
    # STATIC rollout raises on ALL THREE (none lacks a pelvis descent / has the onset it needs), so it
    # scores NONE of them — that is the round-15 protected case. A surgical evader raises on the ONE it
    # would farm while SCORING the others. So: gameable iff ≥1 verdict-probe RAISED and ≥1 was SCORED.
    _VERDICT_PROBES = frozenset({"do_nothing_upright", "jitter_in_place", "velocity_peak_ref"})
    raised_vfp: set[str] = set()
    scored_vfp: set[str] = set()

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
            if name in _VERDICT_PROBES:
                # §round-30/31 UNIFIED probe-raise fail-open closure: a raise on a verdict-feeding
                # probe is tracked; the verdict below fails CLOSED iff this raise is SURGICAL (some
                # OTHER verdict-probe was scored). round-30 also records the reference denominator as
                # 0.0 so the ratio check still trips on a normally-scored high jitter.
                raised_vfp.add(name)
                if loser.get("reference_only"):
                    ref_scores[name] = 0.0
            continue
        s = _gameable_score(s)   # §round-7: a NaN/inf hack score → GAMEABLE (fail-closed)
        losers_scored += 1
        entry = {"name": name, "channel": loser.get("channel"), "score": round(s, 4)}
        if name in _VERDICT_PROBES:
            scored_vfp.add(name)
        if loser.get("reference_only"):
            # §round-26: a paired REFERENCE probe (velocity_peak_ref) — scored for the
            # velocity-floor verdict check below, but NEVER ceiling-compared (a genuine
            # high-intensity metric legitimately scores it high) and never a floor/worst probe.
            entry["reference_only"] = True
            ref_scores[name] = s
            rec["required_losers"].append(entry)
            continue
        # §round-21: a STRICT per-loser floor for a pure-idle probe — it performs NONE of the
        # goal, so any non-trivial credit (≥ max_frac·competent_ref) is unearned/farm-able even
        # though it is below the 0.5 gaming ceiling. Only applies when there is a positive
        # competence anchor (competent_ref > 0).
        mf = loser.get("max_frac")
        if mf is not None and competent_ref > 0:
            thr = float(mf) * float(competent_ref)
            entry["floor"] = round(thr, 4)
            if s >= thr:
                floor_gamed, floor_name, floor_thresh = True, name, thr
        loser_scores[name] = s
        rec["required_losers"].append(entry)
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
    if author_reason:
        # never-silent: surface an author crash/leak/echo even when the DETERMINISTIC losers
        # carried the verdict (ran=True). §round-19: the firewall now runs the goal-blind
        # losers for kicks too, so an author failure no longer makes the gate inconclusive —
        # but the failure is still recorded for provenance.
        rec["author_note"] = author_reason
    if not rec["ran"]:
        rec["reason"] = author_reason or (
            "adversarial: no renderable gaming archetype — inconclusive, not enforced")
        return rec

    # §round-21: competent_ref ≤ 0 means there is NO usable competence anchor, so the gate
    # cannot judge gameability — return INCONCLUSIVE (not a deny). The live grant path can't
    # reach this (the per-source separation gate forces competent_ref ≥ _TD_SEPARATION_MIN);
    # only the audit-only spec probe can, when the spec fn raises on its reference. (Old bug:
    # a 0 ceiling flagged a clean 0.0-scoring metric as "gameable" with worst_name=None.)
    if float(competent_ref) <= 0.0:
        rec["reason"] = ("adversarial: no usable competence anchor (competent_ref ≤ 0) — "
                         "inconclusive, not enforced")
        rec["ceiling"] = None
        return rec

    ceiling = min(rel_ceil * float(competent_ref), abs_ceil)
    rec["ceiling"] = round(ceiling, 4)
    # §round-26 PEAK discriminator: an additive MEAN-velocity-floor confound pays the tiny-ROM
    # jitter probe ~the same as the high-PEAK velocity_peak_ref probe (its 1−exp(−mean/k)
    # saturates), so it is INSENSITIVE to a >2× peak difference — the farming signature, robust
    # to the confound's FLOOR_v (the ratio is FLOOR_v-invariant). A genuine velocity/kick metric
    # scores velocity_peak_ref well ABOVE jitter. Gate on jitter being above the idle floor too,
    # so a metric that scores BOTH ~0 is not flagged.
    vel_floor_gamed = False
    vfj = loser_scores.get("jitter_in_place")
    vfr = ref_scores.get("velocity_peak_ref")
    if (vfj is not None and vfr is not None
            and vfj >= _LOSER_FLOOR_FRAC * float(competent_ref)
            and vfj >= _VEL_FLOOR_RATIO * vfr):
        vel_floor_gamed = True
        rec["velocity_floor"] = {"jitter": round(vfj, 4), "peak_ref": round(vfr, 4),
                                 "ratio_thresh": _VEL_FLOOR_RATIO}
    # §round-32 [HIGH FALSE REJECT] fix of the round-31 rule: the three verdict probes form a
    # MOTION LADDER — do_nothing_upright (mean|jv|=0) < jitter_in_place (≈1.18) < velocity_peak_ref
    # (≈2.94). The round-31 "≥1 RAISED ∧ ≥1 SCORED" rule was OVER-BROAD: an HONEST onset/phase
    # KICK metric (onset = first frame knee-vel > thresh) legitimately RAISES on the ZERO-motion
    # do_nothing (empty selection → IndexError) while SCORING the moving jitter/ref probes far below
    # every threshold — the same surgical signature, but it is not gaming (its guarded twin GRANTS;
    # it ranks the ladder identically to the honest reference). Split by which probe raised:
    #   • jitter_in_place / velocity_peak_ref raised → ALWAYS evasion. Both are MOTION rollouts an
    #     honest joint-velocity metric SCORES without raising; a pelvis-only honest metric that DOES
    #     raise on them (flat pelvis) also raises on do_nothing (all flat) → scored_vfp is empty
    #     (the round-15 case, excluded below). So a motion-probe raise WITH another probe scored is a
    #     surgical fail-open (round-30 D1 ref-floor, round-31 A1 jitter velocity-floor).
    #   • do_nothing_upright raised → AMBIGUOUS (an honest onset metric has no event to time on a
    #     still rollout). Flag it only with POSITIVE stillness-farm evidence: the metric scores the
    #     lower-motion jitter ≥ the higher-motion ref (motion-ANTI-monotone → it would score the
    #     zero-motion do_nothing ≥ both → BREACH the floor it skipped). The honest onset metric scores
    #     ref > jitter (motion-sensitive → do_nothing would be ~0 → floor passes) → NOT flagged.
    motion_probe_raised = bool(raised_vfp & {"jitter_in_place", "velocity_peak_ref"})
    do_nothing_farm = ("do_nothing_upright" in raised_vfp
                       and vfj is not None and vfr is not None and vfj >= vfr)
    probe_raise_gamed = ((motion_probe_raised or do_nothing_farm)
                         and bool(scored_vfp) and float(competent_ref) > 0)
    if probe_raise_gamed:
        rec["probe_raise"] = {"raised": sorted(raised_vfp), "scored": sorted(scored_vfp),
                              "via": "motion_probe" if motion_probe_raised else "do_nothing_farm"}
    gameable = ((not np.isfinite(worst))   # §round-7: non-finite worst → fail-closed
                or (worst >= rel_ceil * float(competent_ref)) or (worst >= abs_ceil)
                or floor_gamed              # §round-21: a pure-idle probe breached its floor
                or vel_floor_gamed          # §round-26: a mean-velocity floor (peak-insensitive)
                or probe_raise_gamed)       # §round-31: surgical probe-raise verdict evasion
    rec["gameable"] = bool(gameable)
    rec["ok"] = not gameable
    if (gameable and probe_raise_gamed and not (worst >= ceiling)
            and not floor_gamed and not vel_floor_gamed):
        rec["worst_name"] = sorted(raised_vfp)[0]
        rec["reason"] = (
            f"adversarial: probe-raise evasion — the metric RAISED on verdict probe(s) "
            f"{sorted(raised_vfp)} while SCORING {sorted(scored_vfp)} — a surgical raise on a "
            f"single-point loser to skip its floor/velocity-floor verdict (an honest metric scores "
            f"these valid rollouts, or an onset metric raises on ALL of them) — gameable")
    elif gameable and vel_floor_gamed and not (worst >= ceiling) and not floor_gamed:
        rec["worst_name"] = "jitter_in_place"
        rec["reason"] = (
            f"adversarial: velocity-floor — jitter_in_place scored {vfj:.3f} ≥ "
            f"{_VEL_FLOOR_RATIO:.0%} of the high-peak velocity_peak_ref ({vfr:.3f}), i.e. the "
            f"metric pays a flat MEAN-velocity floor insensitive to peak intensity — gameable")
    elif gameable and floor_gamed and not (worst >= ceiling):
        # the deny is carried by the per-loser FLOOR, not the upper ceiling — name it precisely.
        fscore = next((l["score"] for l in rec["required_losers"]
                       if l.get("name") == floor_name), worst)
        rec["worst_name"] = floor_name
        rec["reason"] = (
            f"adversarial: pure-idle probe {floor_name!r} scored {fscore:.3f} "
            f"≥ floor {floor_thresh:.3f} ({_LOSER_FLOOR_FRAC:.0%} of competence "
            f"{competent_ref:.3f}) — metric pays unearned credit for doing nothing")
    elif gameable:
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
    valid_ladders: list = []     # §round-20: full authored rung lists (for the crouch-span check)
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
        valid_ladders.append(list(rungs))                      # §round-20 crouch-span check
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
            # §round-17: whether the still-upright / down losers are ON-goal (a balance/lie
            # task) is decided from the blind AUTHORED ladder's top-rung posture, NOT a brittle
            # goal-keyword classifier (which had a new token gap every round).
            ladder_sh, ladder_td = _ladder_posture(valid_top_specs)
            # §round-20 [HIGH FALSE GRANT] fix: the top-rung posture alone conflates a HOLD
            # with a TRANSITION-into-that-posture. A crouch/sit→stand transition has a
            # held-standing TOP rung (static_hold=True per-rung) but do_nothing (already
            # standing, never rose) is OFF-goal — suppress static_hold when a STRICT MAJORITY
            # of valid ladders contain a crouched (low base_height) rung. A genuine balance
            # ladder keeps every rung at nominal height, so it is unaffected.
            if ladder_sh and valid_ladders:
                crouched = sum(1 for L in valid_ladders if _ladder_has_crouched_rung(L))
                if crouched * 2 > len(valid_ladders):
                    ladder_sh = False
            # §round-21 [HIGH FALSE GRANT] / §round-22→23 fix: a blind author can emit a plausible
            # postural-STABILITY ladder (rungs graded by uprightness, held-upright top) for an
            # ACTIVE gesture goal ("salute", "wave your arm"); the top rung passes the per-rung
            # static-hold test and dropping do_nothing then GRANTS a posture/height confound (#6).
            # The keyword classifier is the ONLY goal-aware signal, and it is incomplete in BOTH
            # directions: round-22 tried "VETO only on a positive active verb" → a gesture goal
            # whose verb is off the list (salute/flap/wiggle/tap/…) false-GRANTED (round-23). So
            # drop the posture losers ONLY on POSITIVE still-hold evidence (the SAFE direction: a
            # miss false-REJECTS → observe-only, never a gate-weakening false-grant — mission
            # invariant). The keyword lists are broadened (round-23) to recover accept rate; the
            # DURABLE fix for the residual balance false-rejects is goal-joint scoping
            # (REQUIRED_JOINT_ROLES), tracked as a separate increment. The round-20 crouched-rung
            # suppression still independently guards a crouch/sit→stand transition.
            if ladder_sh and not _goal_is_static_hold(behavior_goal):
                ladder_sh = False
            # §round-27 [HIGH FALSE GRANT] fix: the SIBLING of the ladder_sh guard above — the
            # round-17/20 work hardened the static_hold side (crouched-rung + goal-text backstops)
            # but left ladder_td flowing UNGUARDED to general_required_losers. A blind author can
            # mis-render a RETURNS-UP goal ("squat down then jump straight up") as a descent-ENDING
            # ladder; its top rung passes _spec_is_terminal_down, so collapse_and_stay_down is
            # dropped and a drop-to-floor-and-stay confound GRANTS (B1, reproduced). Apply the same
            # goal-text backstop: an explicitly returns-up / non-terminal goal KEEPS the
            # collapse_and_stay_down loser. SAFE direction (mirrors ladder_sh): a keyword
            # false-negative on a genuine lie/rest goal merely KEEPS the loser → observe-only
            # false-reject, never a gate-weakening false-grant.
            if ladder_td and not _goal_is_terminal_down(behavior_goal):
                ladder_td = False
            # §round-32 [CRITICAL FALSE GRANT] fix: derive whether the goal TRAVELS from the blind
            # ladder's commanded base speed (anti-collusion-safe), to decide whether the
            # walk_away_upright travel probe is on-goal (locomotion → drop) or off-goal (stationary
            # → keep, catching the additive horizontal-travel farm).
            # §round-33/34: the blind ladder's commanded travel is the AUTHORITATIVE, anti-collusion-
            # safe signal — when it travels, the goal IS locomotion and walk_away must be dropped (else
            # an honest locomotion metric scores the probe ≥ ceiling = HARD deny). §round-34 [HIGH
            # FALSE GRANT] fix: the round-33 `or _goal_is_locomotion(...)` let a POLYSEMOUS locomotion
            # token in a STATIONARY goal ("slide a puck across the table", "pace your breathing",
            # "shuffle the deck") DROP walk_away on a non-traveling ladder → re-opened the round-32
            # additive horizontal-travel farm. The keyword can ONLY ADD a (wrong) drop, never recover a
            # missed one beyond what the ladder already encodes — so on the LIVE path TRUST THE LADDER
            # ALONE (the round-33 sidestep false-reject is fixed by the ladder, which DOES travel for a
            # genuine sidestep goal — the keyword was never load-bearing there). The keyword stays only
            # as the no-ladder fallback INSIDE general_required_losers (direct/test callers).
            ladder_travels = _ladder_travels(valid_ladders)
            # §round-33/34 [same fix]: trust the ladder's commanded hops for the jump-goal drop signal
            # (a polysemous jump word in a stationary goal must not drop hop_in_place_upright).
            ladder_hops = _ladder_hops(valid_ladders)
            if fam == "kick" and adversarial_required_losers:
                # opt-in breadth: the DEDICATED kick losers (WITH foot_pos_b direction
                # channel that render_rung can't synthesize).
                req_losers = kick_required_losers(names, behavior_goal, robot_hint)
                sc = list(_KICK_SCORED_CHANNELS)
            else:
                # §round-13/19 FALSE-GRANT fix: the general goal-blind losers (do-nothing /
                # jitter / collapse-and-stay / floor-thrash) run ALWAYS for a novel grant —
                # incl. a novel KICK on the default path (the old `fam=="kick"` branch set
                # req_losers=None there → firewall OFF → a posture/velocity confound granted).
                # A real metric scores them ~0; a confound scores one at/above the ceiling.
                req_losers = general_required_losers(
                    names, behavior_goal, static_hold=ladder_sh, terminal_down=ladder_td,
                    travels=ladder_travels, hops=ladder_hops)
                sc = list(_GENERAL_SCORED_CHANNELS)
            # §round-19: record the obligation for any channel a loser actually covers (the
            # terminal-down floor-thrash loser adds a 'stillness' channel) so coverage_gaps
            # stays honest — a probe that runs but is unlisted would read as an unmet gap.
            if sc is not None and req_losers:
                for ch in (l.get("channel") for l in req_losers):
                    if ch and ch not in sc:
                        sc.append(ch)
            # The LLM breadth pass is opt-in (`adversarial`); the deterministic losers
            # always run, so skipping it never fails the firewall open.
            adv = adversarial_archetype_gate(
                gen_fn, roles, names, competent_ref, client=client, model=model,
                base_payload=base_payload, metric_src=metric_src,
                n_archetypes=adversarial_n,
                required_losers=req_losers, scored_channels=sc,
                author=adversarial)
            # §round-35 FIX A: goal-channel scoping via off-goal-perturbation invariance — the
            # DURABLE close of the additive-SUM / multiplicative off-goal-channel class (rounds
            # 32-34 whack-a-mole). Records `scope` on the adv result; a rise → gameable.
            scope = off_goal_perturbation_verdict(gen_fn, roles, valid_ladders, names)
            if adv is not None:
                adv["scope"] = scope
                if scope.get("gameable") and not adv.get("gameable"):
                    adv["gameable"] = True
                    adv["ok"] = False
                    # §round-37: assign directly — adv['reason'] is pre-seeded to None by the gate, so
                    # setdefault was a no-op and the scope reason was lost (round-36 cosmetic finding).
                    if not adv.get("reason"):
                        if scope.get("goal_joint_insensitive"):
                            adv["reason"] = (
                                f"scope: the metric is INSENSITIVE to its goal joints — a policy that "
                                f"STILLS the goal's {scope.get('n_goal_joints')} joints (doing only the "
                                f"on-goal root/posture {scope.get('on_root') or 'no'}-root) still scores "
                                f"{scope.get('gneut')} ≥ the gaming ceiling {scope.get('ceiling')} "
                                f"(comp {scope.get('comp')}); it reads a root/posture channel instead of "
                                f"the goal — a pelvis-bob/dip farm games it — gameable")
                        elif scope.get("on_goal_char"):
                            adv["reason"] = (
                                f"scope: the metric reads only the goal channel's RANGE, not its "
                                f"required SPEED — under a goal-joint SLOW-DOWN (same ROM, ~zero "
                                f"velocity) it retained {scope.get('slow_retained')} of competence "
                                f"(a burst/velocity metric drops); a slow large-ROM sweep games it "
                                f"on this velocity-characterized goal — gameable")
                        else:
                            adv["reason"] = (
                                f"scope: the metric REWARDS off-goal channels — score moved "
                                f"comp {scope.get('comp')} / +off-goal {scope.get('pert')} / "
                                f"−off-goal {scope.get('neut')} when off-goal joints/root (outside "
                                f"the goal's {scope.get('n_goal_joints')} goal joints / "
                                f"{scope.get('on_root') or 'no'}-root) were perturbed on a competent "
                                f"reference; a goal-scoped metric is invariant — gameable")
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
