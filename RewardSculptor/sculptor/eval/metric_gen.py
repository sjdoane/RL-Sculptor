"""Generate + review an auto objective metric from a behavior goal
(§Ship 35). Mirrors the eval/eureka.py LLM pattern (sample → strip fences)
and the diagnose.py structured-review pattern (messages.parse).

Pipeline: GENERATE (LLM, physical-quantity metric) → VALIDATE (the
must-have gates in metric_validate) → on failure, REGENERATE with the
gate failures fed back (bounded attempts) → independent REVIEW (a fresh
LLM context that never saw the reward) → record a verdict + the
archetype scores to `meta.json`. A metric is `accepted` only if it both
passes validation AND the reviewer approves. Acceptance does NOT grant
steer-rights — that requires calibration (metric_calibration); accepted
metrics run observe-only until calibrated.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from pydantic import BaseModel

from sculptor.eval.metric_validate import validate_generated_metric
from sculptor.eval.robot_manifest import robot_joint_names
from sculptor.llm import (
    llm_log_dir,
    log_llm_call,
    model_for,
    response_text_blocks,
    review_panel_models,
    set_llm_log_dir,
)
from sculptor.prompts import load_prompt
from sculptor.world.channels import ChannelCatalog, resolve_channel_catalog

def _build_reference_signature_block(
    references: list[tuple[str, dict]],
) -> tuple[str, dict[str, Any]]:
    """§REFERENCE_TRAJECTORY_PLAN §7: render `references` (a list of
    `(clip_id, loaded clip dict)` pairs) into a `REFERENCE MOTION
    SIGNATURE` markdown block for the authoring prompt, plus the raw
    `{clip_id: signature}` dict recorded on the result.

    A single reference whose `kinematic_signature` call raises (a
    corrupt/invalid clip slipping past the caller) is skipped with its
    error recorded under `"_error"` in the returned dict — never lets a
    bad reference crash generation; the block still renders every clip
    that DID summarize successfully."""
    from sculptor.refs.convert import kinematic_signature

    signatures: dict[str, Any] = {}
    lines = [
        "# REFERENCE MOTION SIGNATURE",
        "The competent motion looks like THIS — write a metric scoring "
        "these numbers high; ground thresholds in these real values, do "
        "not guess.",
        "",
    ]
    for clip_id, clip in references:
        try:
            sig = kinematic_signature(clip)
        except Exception as e:  # noqa: BLE001 — one bad clip must not crash gen
            signatures[clip_id] = {"_error": f"{type(e).__name__}: {e}"}
            continue
        signatures[clip_id] = sig
        lines.append(f"## clip_id: {clip_id}")
        lines.append(json.dumps(sig, indent=2, default=str))
        lines.append("")
    # §D24 F3: two authoring rules closing the D23/D22 reference-anchored
    # failure classes, in the same "hard-won rejection cause" style as the
    # base prompt's TIME-SERIES HYGIENE section (D12). Scoped to the
    # reference block (rendered only when references are attached) since
    # both rules are specifically about scoring a REFERENCE completion
    # correctly regardless of when/how it is held.
    lines.append("## RELATIVE-TIME RULE (hard-won rejection cause)")
    lines.append(
        "- NEVER assume absolute frame counts, absolute timestamps, or that "
        "the goal occurs at a fixed FRACTION of the episode length. A "
        "rollout may reach the goal EARLY and hold it there for the rest of "
        "the episode — the metric MUST score a reference-prefix-then-hold "
        "trajectory (the goal reached, then the terminal state sustained) "
        "AS HIGHLY as the reference itself. End-window reads (\"is the LAST "
        "N frames' state near the goal\") are fine; \"the transition must "
        "happen at t=X\" or \"phase Y occupies frames [a, b)\" is not — a "
        "metric will be scored on exactly this reach-then-hold shape and "
        "rejected if it does not rank it with the reference.")
    lines.append("")
    lines.append("## HOLD-GOAL RULE (hard-won rejection cause)")
    lines.append(
        "- When the goal includes HOLDING a terminal state (a stand, a sit, "
        "a balance), the completion gate MUST ALSO require the START of the "
        "scored window to be AWAY FROM / BELOW that terminal state — never "
        "satisfied by a clip that is already in the goal state from frame 0. "
        "Without this, a rollout that starts (or resets) already in the "
        "terminal pose and never performs the transition scores identically "
        "to one that actually achieved it — the metric must distinguish "
        "'reached the goal' from 'was already there'.")
    lines.append("")
    lines.append("## FAST-COMPLETION RULE (hard-won rejection cause)")
    lines.append(
        "- The policy may complete the transition MUCH FASTER than the "
        "reference — a real rollout finished an 8.1 s human righting span "
        "in ~0.5 s (16x) and a certified metric zeroed it because its "
        "start-state read was a 0.5 s WINDOW MEAN: the completed state "
        "leaked into the 'start' window, killing both the started-away "
        "gate and the change-amplitude floor. Start-state reads must use "
        "the EARLIEST frames only (first 2-5 frames / <= 0.1 s) or, "
        "better, compare against the EVAL START STATE numbers directly — "
        "the episode is GUARANTEED to begin there; measuring 'did it "
        "start away from the goal' over a window the policy can finish "
        "inside is a self-zeroing bug. The metric will be scored on a "
        "16x-sped-then-held variant of the reference and rejected if it "
        "does not rank it with the reference (for reach-and-hold goals).")
    lines.append("")
    return "\n".join(lines), signatures


def _build_eval_reset_block(eval_reset: dict[str, Any]) -> str:
    """§D24 F2: render the stage's ACTUAL certified eval-rollout START
    state (§D17 stage-fixed eval reset, settled via §D20a's
    settle-then-rederive) into a `# EVAL START STATE` authoring block —
    `eval_reset` is the `{"scalars": ..., "settled": bool, "reason":
    ...}` payload `mission_metrics._compute_eval_reset_preview` returns.

    Every certified rollout's episode BEGINS at these numbers, NOT at the
    reference clip's own frame 0 (D23/H2 class: a metric that implicitly
    calibrates a "started low"/"started away from the goal" gate to the
    clip's own initial frame can silently mis-score every rollout from
    iteration one if the two disagree — the exact failure the
    `reference_settled_start` validation gate exists to catch
    mechanically; this block is the authoring-side half of the same fix).

    Returns `""` for a falsy/malformed payload (defensive — this is
    advisory context that must never crash generation) or one whose
    `scalars` carry no usable height offset."""
    scalars = (eval_reset or {}).get("scalars")
    if not isinstance(scalars, dict) or "reset_height_offset_m" not in scalars:
        return ""
    from sculptor.reference import G1_CLASS_STAND_M

    root_z = round(
        G1_CLASS_STAND_M + float(scalars["reset_height_offset_m"]), 4)
    settled = bool(eval_reset.get("settled"))
    payload = {
        "root_z_m": root_z,
        "pitch_rad": scalars.get("reset_pitch_offset_rad"),
        "roll_rad": scalars.get("reset_roll_offset_rad"),
        "settled": settled,
    }
    lines = [
        "# EVAL START STATE",
        "Every certified rollout's EPISODE BEGINS HERE (a stage-fixed "
        "eval-rollout reset) — NOT at the reference clip's own frame 0. "
        "A \"started low\" / \"started away from the goal\" gate MUST "
        "accept THIS start state; do not calibrate a start-window "
        "threshold against the clip's own initial frame if it disagrees "
        "with these numbers — a metric that does will be scored on "
        "exactly this mismatch and rejected.",
        "",
        json.dumps(payload, indent=2, default=str),
    ]
    if not settled:
        reason = eval_reset.get("reason")
        lines.append(
            "(settled=false: physical settling was unavailable at "
            "generation time" + (f" ({reason})" if reason else "") +
            " — these are the derived, UNSETTLED numbers; treat them as "
            "approximate.)")
    return "\n".join(lines)


MODEL_ID = model_for("metric_gen")
#: §truncation fix: adaptive THINKING shares this budget with the code output, and a
#: hard metric's think can use 5-8k tokens — at the old 8000 cap that truncated the
#: code (a confusing downstream "missing compute_spec"). 16000 leaves ample room for
#: thinking AND a complete metric (observed ~7.3-7.5k total per generation).
MAX_TOKENS = 16000
#: §review-feedback retry: how many times a VALIDATION-passing but REVIEWER-VETOED
#: metric is regenerated with the reviewer's concerns fed back before giving up. Bounds
#: the extra (gen + review) cost; the validation-failure feedback loop is separate.
_MAX_REVIEW_RETRIES = 2
_FENCE_RE = re.compile(r"```(?:[A-Za-z]+)?\s*\n(.*?)```", re.DOTALL)


class MetricReview(BaseModel):
    """Independent reviewer's structured verdict. `gaming_exploit` is a CONSTRUCTIVE
    exploit the reviewer names (a concrete degenerate behavior that scores a channel
    ≥ floor while not doing the task); "" = none. A non-empty value forces a reject
    (a 'named exploit but approved' contradiction can never slip through)."""

    approved: bool
    concerns: list[str] = []
    summary: str = ""
    gaming_exploit: str = ""


# ── §Metric-quality laws (LAW 9): VLM metric-review Panel A ───────────────
#: The reviewer model pool — the repo's FIRST multi-model surface. Single vendor,
#: so MODEL-ID diversity is the "cross-family" substitute (Verga PoLL 2404.18796:
#: disjoint juries beat one judge with decorrelated error). All are Active +
#: vision-capable. The metric AUTHOR's exact id (`MODEL_ID`) is excluded from the
#: roster (Panickssery NeurIPS 2024: a model self-prefers its own output).
#: Sourced from the sculptor.llm registry (RS_MODEL_REVIEW_PANEL overrides);
#: the defaults there are author-disjoint from model_for("metric_gen") by
#: construction, and `_reviewers_from_models` re-excludes the ACTUAL author
#: at call time as the runtime backstop.
REVIEW_PANEL_MODELS = tuple(review_panel_models())
DEFAULT_REVIEW_MODELS = REVIEW_PANEL_MODELS   # convenience alias for callers

#: The lens that judges goal-match + naturalness from rollout KEYFRAMES via a VLM
#: (the others are text-only). It has no keyframes at metric-GENERATION time (pre-
#: rollout), so at launch-gen it always SKIPS and is excluded from quorum; its live
#: call site is a future POST-rollout re-review (where keyframes exist).
_VLM_LENS = "naturalness"
_LENS_ORDER = ("completeness", "measurability", "over_restriction",
               "consistency", _VLM_LENS)

#: Per-lens FOCUS appendix layered on the shared `review_objective_metric` rubric
#: (each reviewer sees the full rubric + one focus). Diversified lenses → decorrelated
#: error; the base rubric keeps every reviewer fail-toward-reject (when in doubt,
#: do not approve).
_LENSES: dict[str, str] = {
    "completeness":
        "FOCUS LENS: completeness. For EACH scored channel, name a concrete gaming "
        "policy (a constructive exploit) that scores it >= floor while NOT achieving "
        "the goal — including an UNENUMERATED hack beyond the listed archetypes. If "
        "any is plausible, set gaming_exploit and reject.",
    "measurability":
        "FOCUS LENS: measurability. Is each channel COMPUTABLE from the declared / "
        "allowed physical arrays, or is it a silent proxy for a signal that is not "
        "present (LAW 6)? A channel that substitutes a magnitude for an absent signal "
        "(e.g. foot direction) must be observe-only — reject if it is scored anyway.",
    "over_restriction":
        "FOCUS LENS: over-restriction. Does any directional / postural / support gate "
        "fire WITHOUT a declared frame field, so it would false-reject a legitimate "
        "handstand, backflip, deliberately-rearward mule-kick, or single-support "
        "variant (LAW 0)? Unresolved frame fields must ABSTAIN, not default.",
    "consistency":
        "FOCUS LENS: composition. Is the score EXACTLY a sharp completion gate times a "
        "min of saturating channels — with NO weighted sum, NO fractional partial-"
        "credit product, NO peak/median ratio, and NO unbounded/extremal term a single-"
        "frame spike can inflate (LAW 1/8)? Reject any other composition. (A SEPARATE "
        "`progress_score` subcomponent — a min of the same channels WITHOUT the gate, "
        "for search ranking only — is expected and fine UNLESS it feeds spec_score, "
        "uses a non-min composition, or scores a dead-still policy above ~0.)",
    _VLM_LENS:
        "FOCUS LENS: naturalness & goal-match. From the KEYFRAMES, judge whether the "
        "depicted motion actually ACHIEVES the goal (goal-match) and is physically "
        "plausible (no joint-limit-violating flail). Separately, in the SOURCE, confirm "
        "naturalness is a SEPARATE channel, not fused into correctness (LAW 7). If no "
        "keyframes are provided you cannot assess goal-match from images — say so.",
}


def _reviewers_from_models(
    models: list[str], author_model: str,
) -> list[tuple[str, str]]:
    """Assign the fixed lens roster round-robin over `models` MINUS the author's
    exact id (blinding / self-preference). Falls back to the given models only if
    excluding the author would empty the pool (defensive)."""
    pool = [m for m in models if m != author_model] or list(models)
    if not pool:
        return []
    return [(pool[i % len(pool)], lens) for i, lens in enumerate(_LENS_ORDER)]


def _review_one(
    client: Any, model: str, lens: str, goal: str, source: str,
    archetype_scores: dict, keyframes: Optional[list] = None,
    selectivity: Optional[dict] = None,
) -> dict[str, Any]:
    """One reviewer (model × lens). Returns
    `{model, lens, status: approved|veto|skip|error, reason, concerns, gaming_exploit}`.
    The VLM lens with no keyframes SKIPS (excluded from quorum). A crashed call is
    `error` (no-evidence — counts against quorum but is NEVER a veto). A non-empty
    `gaming_exploit` forces a veto even if the model returned approved=True.

    `selectivity` (the validate selectivity-probe scores, present only on a NOVEL-goal
    vacuous pass) is surfaced so the reviewer sees the metric IS selective — otherwise
    it sees an all-zero fixed battery and wrongly flags "can't confirm not near-constant"."""
    is_vlm = lens == _VLM_LENS
    if is_vlm and not keyframes:
        return {"model": model, "lens": lens, "status": "skip",
                "reason": f"{lens} (VLM) lens: no keyframes at this stage — skipped",
                "concerns": [], "gaming_exploit": ""}
    appendix = _LENSES.get(lens, "")
    # Stable rubric in a cacheable system prefix; volatile per-metric JSON in user.
    system = [{"type": "text", "text": load_prompt("review_objective_metric"),
               "cache_control": {"type": "ephemeral"}},
              {"type": "text", "text": appendix}]
    pay: dict[str, Any] = {"behavior_goal": goal, "metric_source": source,
                           "archetype_scores": archetype_scores}
    if selectivity is not None:
        pay["selectivity_probe"] = selectivity
    payload = json.dumps(pay, indent=2, default=str)
    if is_vlm:
        from sculptor.diagnose import _encode_image   # reuse the encoder (LAW 9)

        user_content: Any = [{"type": "text", "text": payload
                              + "\n\nKEYFRAMES (evenly-spaced from the best episode):"}]
        for kf in keyframes or []:
            user_content.append(_encode_image(kf))
        user_content.append({"type": "text", "text": "Emit the structured review now."})
    else:
        user_content = payload
    try:
        resp = client.messages.parse(
            model=model, max_tokens=4000, thinking={"type": "adaptive"},
            system=system, messages=[{"role": "user", "content": user_content}],
            output_format=MetricReview)
        log_llm_call(
            "metric_review", model,
            system="\n\n".join(str(b.get("text", "")) for b in system),
            user=payload,   # image blocks elided (keyframes live on disk)
            response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None),
            meta={"lens": lens, "juror": model,
                  "n_keyframes": len(keyframes or []) if is_vlm else 0})
        r = resp.parsed_output
        approved = bool(r.approved)
        exploit = str(getattr(r, "gaming_exploit", "") or "")
        concerns = list(r.concerns)
        if exploit and approved:   # a named exploit forces a reject (no contradiction)
            approved = False
            concerns = [f"named a gaming exploit but returned approved — treating as "
                        f"reject: {exploit}", *concerns]
        return {"model": model, "lens": lens,
                "status": "approved" if approved else "veto",
                "reason": ((r.summary or "approved") if approved
                           else (exploit or "; ".join(concerns) or "rejected")),
                "concerns": concerns, "gaming_exploit": exploit}
    except Exception as e:  # noqa: BLE001 — a crashed reviewer is no-evidence, NOT a veto
        return {"model": model, "lens": lens, "status": "error",
                "reason": f"review call failed: {type(e).__name__}: {e}",
                "concerns": [], "gaming_exploit": ""}


def _review_metric_panel(
    client: Any, goal: str, source: str, archetype_scores: dict, *,
    reviewers: list[tuple[str, str]], keyframes: Optional[list] = None,
    selectivity: Optional[dict] = None,
) -> dict[str, Any]:
    """§Ship 55 (LAW 9): a diversified, blinded review PANEL. Each reviewer (model ×
    lens) votes; aggregation is asymmetric-veto with an ABSOLUTE quorum FLOOR:

        approved = (n_eligible >= 1) and (n_approve >= ceil(n_eligible/2)) and not vetoed

    where `vetoed` = ANY returned reviewer rejected (a constructive `gaming_exploit`
    is the strongest reject), and CRASHES count AGAINST the quorum (they do not
    shrink the denominator) — so a thin lone survivor cannot carry a panel and an
    all-fail panel ends NOT-approved (fail-closed overall). The no-keyframe VLM skip
    is excluded from `n_eligible`. Returns `{review, panel, veto_by, quorum,
    n_eligible, n_approve, models, lenses}`; `review` is EXACTLY
    `{approved, concerns, summary}` (the byte-stable UI contract) — provenance lives
    in the sibling keys, never nested in `review`. Never-silent: a veto / quorum
    miss names the (model, lens, reason) in `review.concerns`."""
    panel = [_review_one(client, m, lens, goal, source, archetype_scores, keyframes,
                         selectivity)
             for (m, lens) in reviewers]
    eligible = [p for p in panel if p["status"] != "skip"]
    n_elig = len(eligible)
    n_approve = sum(1 for p in panel if p["status"] == "approved")
    vetoes = [p for p in panel if p["status"] == "veto"]
    quorum = -(-n_elig // 2)                          # ceil(n_elig / 2)
    approved = (n_elig >= 1) and (n_approve >= quorum) and (not vetoes)

    concerns: list[str] = []
    for v in vetoes:                                  # name every veto (never-silent)
        concerns.append(f"[veto {v['model']}/{v['lens']}] {v['reason']}")
    if not vetoes and not approved:                   # quorum miss / no eligible
        concerns.append(
            f"[quorum] only {n_approve}/{n_elig} eligible reviewers approved "
            f"(need {quorum})" if n_elig
            else "[panel] no eligible reviewer returned a verdict — inconclusive")
    for p in panel:                                   # surface per-reviewer concerns
        for c in p.get("concerns", []):
            concerns.append(f"[{p['model']}/{p['lens']}] {c}")

    summary = (f"panel: {n_approve}/{n_elig} approve, {len(vetoes)} veto, "
               f"{len(panel) - n_elig} skip -> {'approved' if approved else 'rejected'}")
    return {
        "review": {"approved": approved, "concerns": concerns, "summary": summary},
        "panel": panel,
        "veto_by": [f"{v['model']}/{v['lens']}" for v in vetoes],
        "quorum": quorum, "n_eligible": n_elig, "n_approve": n_approve,
        "models": sorted({m for m, _ in reviewers}),
        "lenses": [lens for _, lens in reviewers],
    }


def _strip_code(text: str) -> str:
    blocks = _FENCE_RE.findall(text or "")
    if blocks:
        code = [b for b in blocks if "def compute_spec" in b]
        return (code[0] if code else max(blocks, key=len)).strip() + "\n"
    return (text or "").strip() + "\n"


def _sample_source(client: Any, system_prompt: str, user_content: str,
                   *, model: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    # Provenance BEFORE the truncation gate so truncated generations are
    # archived too (they explain a failed attempt).
    log_llm_call(
        "metric_gen", model, system=system_prompt, user=user_content,
        response_text=response_text_blocks(resp),
        usage=getattr(resp, "usage", None))
    # A max_tokens-truncated response (adaptive thinking ate the budget) yields
    # INCOMPLETE code — surface it explicitly here so it fails as "truncated", not as
    # a baffling downstream "missing compute_spec", and so a retry can recover.
    if getattr(resp, "stop_reason", None) == "max_tokens":
        raise RuntimeError(
            f"generation truncated at max_tokens={MAX_TOKENS} — the model's thinking "
            f"consumed the token budget and the metric code is incomplete")
    chunks = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    return _strip_code("\n".join(chunks))


# ── §best-of-N: sample N candidates, select the most-discriminating valid one ──
#: Per-candidate FRAMING nudges (varied composition emphasis) — the ONLY
#: decorrelation lever. Temperature is NOT varied: the generator runs with extended
#: THINKING (adaptive), and the Anthropic API rejects any temperature != 1.0 when
#: thinking is enabled (a 400 that fails the call instantly). So candidates differ by
#: framing + the model's inherent temp-1.0 stochasticity; all produce the SAME
#: contract (compute_spec), only the angle differs, so the offline discriminator picks
#: the sharpest grader. Index 0 is the empty (default) framing.
_BON_FRAMINGS = (
    "",
    "Variation: make the completion gate as SHARP as possible — a crisp pass/fail "
    "boundary on every prerequisite, with no partial credit leaking through.",
    "Variation: emphasize smooth, well-SEPARATED channel saturation curves so the "
    "score rises monotonically across increasing competence.",
    "Variation: use the MINIMAL set of channels that fully pin the goal, each "
    "saturating independently, combined as a min (the weakest link gates).",
)


def _best_of_n(
    client: Any, system_prompt: str, base_user: str, *,
    behavior_goal: str, robot_hint: Optional[str], robot_names: Optional[list],
    out_dir: Path, metric_path: Path, n: int, model: str,
    emit: Callable[[dict[str, Any]], None],
    references: Optional[list[tuple[str, dict]]] = None,
    eval_reset: Optional[dict[str, Any]] = None,
    channel_catalog: ChannelCatalog | None = None,
) -> tuple[bool, str, Optional[dict[str, Any]], list[dict[str, Any]], Optional[int]]:
    """Sample `n` candidate metrics (decorrelated by FRAMING — temperature stays 1.0,
    required by the generator's extended thinking), validate each, and
    select the VALID one with the highest OFFLINE discrimination (`graded_discrimination`);
    ties keep the FIRST valid (lowest index — today's first-valid behavior). All-invalid
    → reject (run stays blind), exactly as the single path. Returns
    `(passed, source, validation, candidate_records, selected_index, ranked_valid)` where
    `ranked_valid` is the valid candidates (each `{candidate, src, val, discrimination}`)
    sorted best-first, for the caller's review-in-order. Never raises on a bad candidate
    (an API/validation/score failure is recorded, never fatal)."""
    from sculptor.eval.metric_validate import discrimination_of_metric

    n = max(2, int(n))
    cands: list[dict[str, Any]] = []
    for i in range(n):
        framing = _BON_FRAMINGS[i % len(_BON_FRAMINGS)]
        emit({"stage": "generating", "attempt": i + 1, "max": n,
              "message": f"Sampling candidate metric {i + 1}/{n} (best-of-{n})…"})
        user = base_user + (f"\n\n{framing}" if framing else "")
        try:
            src = _sample_source(client, system_prompt, user, model=model)
        except Exception as e:  # noqa: BLE001 — one candidate's API failure is not fatal
            cands.append({"candidate": i, "api_error": f"{type(e).__name__}: {e}"})
            continue
        cpath = out_dir / f"candidate_{i}.py"
        cpath.write_text(src, encoding="utf-8")
        emit({"stage": "validating", "attempt": i + 1, "max": n,
              "message": f"Validating candidate {i + 1}/{n}…"})
        val = validate_generated_metric(
            src, cpath, behavior_goal=behavior_goal, robot_hint=robot_hint,
            robot_joint_names=robot_names, references=references,
            eval_reset=eval_reset, channel_catalog=channel_catalog)
        rec: dict[str, Any] = {"candidate": i, "ok": bool(val["ok"]),
                               "reasons": val["reasons"], "_src": src, "_val": val}
        if val["ok"]:
            disc = discrimination_of_metric(
                cpath, val.get("required_roles") or [],
                channel_catalog=channel_catalog)
            rec["discrimination"] = float(disc.get("score", 0.0))
            rec["discrimination_detail"] = disc
        cands.append(rec)

    valid = [c for c in cands if c.get("ok")]
    clean = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands]

    def _cleanup() -> None:
        # Non-winner candidate files are debug-only; metric.py + meta.json are the
        # contract. Best-effort unlink (never fatal).
        for c in cands:
            try:
                (out_dir / f"candidate_{c['candidate']}.py").unlink()
            except OSError:
                pass

    if not valid:
        # All-invalid → reject (run stays blind). Persist the LAST sampled source to
        # metric.py (mirrors the retry path recording the last attempt) so metric_path
        # points at a real file. Surface a SPECIFIC reason (never-silent): aggregate
        # every candidate's API error / validation reasons so the UI shows WHY all N
        # failed (e.g. a 400 model/auth error) instead of a bare "rejected".
        last = next((c for c in reversed(cands) if "_src" in c), None)
        src = last["_src"] if last else ""
        why: list[str] = []
        for c in cands:
            if c.get("api_error"):
                why.append(f"candidate {c['candidate'] + 1}: API error: {c['api_error']}")
            else:
                for r in (c.get("reasons") or [])[:2]:
                    why.append(f"candidate {c['candidate'] + 1}: {r}")
        if last and last.get("_val"):
            val: Optional[dict[str, Any]] = {**last["_val"], "reasons": why or last["_val"].get("reasons", [])}
        else:
            # every candidate API-errored (no validation ran) — build a minimal record
            # so the caller/UI still gets the reasons rather than a None.
            val = {"ok": False, "reasons": why or ["all candidates failed to generate"],
                   "archetype_scores": {}}
        if src:
            metric_path.write_text(src, encoding="utf-8")
        _cleanup()
        return False, src, val, clean, None, []

    # Rank valid candidates by discrimination (tie → lowest candidate index); the top
    # is the default winner (written to metric.py), and the full RANKED list is returned
    # so the caller can review them IN ORDER (the offline discriminator can't predict the
    # LLM reviewer, so a flawed top candidate shouldn't sink the run when a sibling passes).
    ranked = sorted(valid, key=lambda c: (-c["discrimination"], c["candidate"]))
    best = ranked[0]
    metric_path.write_text(best["_src"], encoding="utf-8")
    emit({"stage": "selecting", "max": n,
          "message": f"Selected candidate {best['candidate'] + 1}/{n} "
                     f"(discrimination {best['discrimination']:.3f}) of "
                     f"{len(valid)} valid."})
    _cleanup()
    ranked_pub = [{"candidate": c["candidate"], "src": c["_src"], "val": c["_val"],
                   "discrimination": c["discrimination"]} for c in ranked]
    return True, best["_src"], best["_val"], clean, best["candidate"], ranked_pub


def _review_metric(
    client: Any, model: str, goal: str, source: str,
    archetype_scores: dict, selectivity: Optional[dict] = None,
) -> dict[str, Any]:
    system_prompt = load_prompt("review_objective_metric")
    pay: dict[str, Any] = {
        "behavior_goal": goal,
        "metric_source": source,
        "archetype_scores": archetype_scores,
    }
    if selectivity is not None:   # NOVEL-goal vacuous pass: prove selectivity to the reviewer
        pay["selectivity_probe"] = selectivity
    user = json.dumps(pay, indent=2, default=str)
    try:
        resp = client.messages.parse(
            model=model,
            max_tokens=4000,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=[{"role": "user", "content": user}],
            output_format=MetricReview,
        )
        log_llm_call(
            "metric_review", model, system=system_prompt, user=user,
            response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None))
        r = resp.parsed_output
        approved = bool(r.approved)
        exploit = str(getattr(r, "gaming_exploit", "") or "")
        concerns = list(r.concerns)
        if exploit and approved:   # §LAW 9: a named exploit forces a reject (no
            approved = False       # 'named exploit but approved' contradiction) —
            concerns = [f"named a gaming exploit but returned approved — treating "
                        f"as reject: {exploit}", *concerns]  # mirrors _review_one
        return {"approved": approved, "concerns": concerns, "summary": r.summary}
    except Exception as e:  # noqa: BLE001 — a failed review is a non-approval
        return {"approved": False,
                "concerns": [f"review call failed: {type(e).__name__}: {e}"],
                "summary": ""}


def generate_objective_metric(
    behavior_goal: str,
    out_dir: Path | str,
    *,
    robot_hint: Optional[str] = None,
    client: Any = None,
    model: str = MODEL_ID,
    max_attempts: int = 3,
    n_candidates: int = 1,
    review: bool = True,
    review_models: Optional[list[str]] = None,
    review_keyframes: Optional[list] = None,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    references: Optional[list[tuple[str, dict]]] = None,
    eval_reset: Optional[dict[str, Any]] = None,
    channel_catalog: (
        ChannelCatalog | Mapping[str, Any] | Path | str | None
    ) = None,
) -> dict[str, Any]:
    """Generate, validate, (regenerate on failure,) and review an objective
    metric for `behavior_goal`. Writes `metric.py` + `meta.json` to
    `out_dir` and returns the full record. NEVER raises on a bad
    candidate — a rejected metric is recorded with `accepted=False`.

    §Ship 40: `on_event(ev)` (optional) is called with a `{stage, attempt,
    max, message}` dict at each pipeline step so the UI can show LIVE progress
    (this is a 1-2 min, multi-LLM-call op). Never fatal — a raising callback
    is swallowed.

    §Ship 55 (LAW 9): `review_models` (None → the unchanged single-reviewer path,
    BYTE-IDENTICAL) selects the VLM Panel A — a diversified, blinded, multi-model
    review panel (`_review_metric_panel`) whose aggregate verdict gates acceptance.
    `review_keyframes` feeds the VLM goal-match/naturalness lens (none at metric-gen
    time → that lens skips). Panel provenance is recorded under `review_panel`; the
    `review` key stays exactly `{approved, concerns, summary}`.

    §best-of-N: `n_candidates` (default 1 → BYTE-IDENTICAL single-shot-with-retry)
    samples N candidates (varied temperature/framing), validates each, and selects the
    VALID one with the highest OFFLINE discrimination (`graded_discrimination` — a
    deterministic graded competence ladder); ties keep the first valid; all-invalid
    rejects (run stays blind), as today. The selected candidate then goes through the
    SAME review. Records `n_candidates`, `candidates`, `selected_candidate`.

    §REFERENCE_TRAJECTORY_PLAN §7: `references` (optional — a list of
    `(clip_id, loaded clip dict)` pairs, default None → BYTE-IDENTICAL
    behavior) grounds generation in real kinematic data. When present: (a)
    a `REFERENCE MOTION SIGNATURE` block (§7 kinematic_signature per clip)
    is injected into the authoring prompt telling the model to ground
    thresholds in these real numbers instead of guessing; (b) it is passed
    through to `validate_generated_metric` so every candidate also
    certifies against the reference-anchored gates (§5); (c) a
    reference-gate failure feeds back into the retry loop's prompt exactly
    like any other validation failure; (d) the result records
    `"references_used"` (the clip ids) and `"reference_signatures"` (the
    rendered signature dict, keyed by clip id — a clip whose signature
    extraction raised records `{"_error": ...}` instead of crashing the
    run).

    §D24 F2: `eval_reset` (optional — the stage's certified eval-reset
    preview, `mission_metrics._compute_eval_reset_preview`'s return
    value; default None → BYTE-IDENTICAL behavior) grounds authoring AND
    validation in the ACTUAL eval-rollout start state (not the reference
    clip's own frame 0): (a) a `# EVAL START STATE` block
    (`_build_eval_reset_block`) is appended to the authoring prompt; (b)
    it is passed through to `validate_generated_metric` on every attempt/
    candidate so the `reference_settled_start` gate (§5) can score a
    settled-start variant; (c) the result records `"eval_reset_preview"`
    verbatim for observability."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=2, timeout=240.0)

    catalog = resolve_channel_catalog(channel_catalog)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # §llm provenance: every generation/review call in this job archives to
    # out_dir/llm_calls.jsonl. The contextvar is last-set-wins, so RESTORE the
    # caller's sink on exit — a backend metric job must not redirect
    # subsequent unrelated calls (sculpt iters keep their own iter_dir sink).
    _prev_llm_log_dir = llm_log_dir()
    set_llm_log_dir(out_dir)
    try:
        metric_path = out_dir / "metric.py"
        # §Ship 49: the ACTUAL robot's joint_names (manifest, keyed by robot_hint)
        # so the required-joint-roles gate can reject a metric whose declared roles
        # don't exist on this robot — before any GPU run. None for an unknown robot
        # (the runtime resolution is then the backstop).
        robot_names = robot_joint_names(robot_hint)
        system_prompt = load_prompt("gen_objective_metric")
        base_user = json.dumps(
            {"behavior_goal": behavior_goal, "robot_hint": robot_hint},
            indent=2, default=str,
        )
        if catalog is not None:
            catalog_context = {
                "channel_catalog_hash": catalog.catalog_hash,
                "allowed_arrays": list(catalog.allowed_metric_arrays()),
                "channel_catalog": catalog.to_dict(),
                "channel_access_rule": (
                    "The objective metric may read every listed channel. "
                    "metric_only channels are held out from reward shaping; "
                    "shared_shaping channels may also inform dense rewards."
                ),
            }
            base_user += "\n\n# EXACT CHANNEL CATALOG\n" + json.dumps(
                catalog_context, indent=2, default=str)
        references_used: list[str] = []
        reference_signatures: dict[str, Any] = {}
        if references:
            sig_block, reference_signatures = _build_reference_signature_block(
                list(references))
            references_used = [clip_id for clip_id, _clip in references]
            base_user = base_user + "\n\n" + sig_block
        if eval_reset:
            eval_reset_block = _build_eval_reset_block(eval_reset)
            if eval_reset_block:
                base_user = base_user + "\n\n" + eval_reset_block

        n_attempts = max(1, max_attempts)
        attempts: list[dict[str, Any]] = []
        source = ""
        validation: Optional[dict[str, Any]] = None

        def _emit(ev: dict[str, Any]) -> None:
            if on_event is not None:
                try:
                    on_event(ev)
                except Exception:  # noqa: BLE001 — progress is advisory, never fatal
                    pass

        candidates: Optional[list[dict[str, Any]]] = None
        selected_candidate: Optional[int] = None
        ranked_valid: list[dict[str, Any]] = []
        passed = False

        if n_candidates and n_candidates > 1:
            # §best-of-N: sample N candidates and select the most-discriminating valid one.
            passed, source, validation, candidates, selected_candidate, ranked_valid = _best_of_n(
                client, system_prompt, base_user,
                behavior_goal=behavior_goal, robot_hint=robot_hint,
                robot_names=robot_names, out_dir=out_dir, metric_path=metric_path,
                n=n_candidates, model=model, emit=_emit, references=references,
                eval_reset=eval_reset, channel_catalog=catalog)
            if not passed:
                # best-of-N samples DIVERSE candidates but does NOT feed validation
                # failures back; if NONE were valid, fall through to the retry-with-
                # feedback loop below (seeded with the aggregated candidate reasons) so
                # best-of-N is NEVER WORSE than single-shot — it adds diversity-selection
                # on top of correction, instead of trading one for the other.
                _emit({"stage": "regenerating", "max": n_attempts,
                       "message": f"All {n_candidates} candidates failed — retrying with "
                                  f"the gate feedback…"})

        # The single-shot-with-feedback retry loop: the only path when n_candidates==1,
        # AND the fallback when best-of-N found no valid candidate (passed is False).
        if not passed:
            for attempt in range(n_attempts):
                _emit({"stage": "generating", "attempt": attempt + 1, "max": n_attempts,
                       "message": f"Generating candidate metric "
                                  f"(attempt {attempt + 1}/{n_attempts})…"})
                user = base_user
                # Feed back ANY prior failure's reasons (an earlier attempt OR, when
                # best-of-N fell through, its aggregated candidate failures). Equivalent
                # to `attempt > 0` for single-shot (validation is None at attempt 0).
                if validation is not None and not validation.get("ok"):
                    user = (
                        base_user
                        + "\n\nThe previous attempt FAILED these validation gates:\n"
                        + json.dumps(validation.get("reasons") or [], indent=2)
                        + "\nFix ALL of them. Output ONLY the corrected module."
                    )
                try:
                    source = _sample_source(client, system_prompt, user, model=model)
                except Exception as e:  # noqa: BLE001 — API failure = failed attempt
                    attempts.append({"attempt": attempt,
                                     "api_error": f"{type(e).__name__}: {e}"})
                    if attempt + 1 < n_attempts:  # §Ship 40 review: only if a retry follows
                        _emit({"stage": "retrying", "attempt": attempt + 1,
                               "max": n_attempts,
                               "message": f"Generation call failed (attempt "
                                          f"{attempt + 1}); retrying…"})
                    continue
                metric_path.write_text(source, encoding="utf-8")
                _emit({"stage": "validating", "attempt": attempt + 1, "max": n_attempts,
                       "message": "Validating (safety / determinism / bounds / "
                                  "non-degeneracy)…"})
                validation = validate_generated_metric(
                    source, metric_path,
                    behavior_goal=behavior_goal, robot_hint=robot_hint,
                    robot_joint_names=robot_names, references=references,
                    eval_reset=eval_reset, channel_catalog=catalog)
                attempts.append({"attempt": attempt, "ok": validation["ok"],
                                 "reasons": validation["reasons"]})
                if validation["ok"]:
                    break
                if attempt + 1 < n_attempts:
                    _emit({"stage": "regenerating", "attempt": attempt + 1,
                           "max": n_attempts,
                           "reasons": (validation.get("reasons") or [])[:2],
                           "message": f"Attempt {attempt + 1} failed validation — "
                                      f"regenerating with the gate feedback…"})

            passed = bool(validation and validation["ok"])
        def _dispatch_review(src: str, val: Optional[dict[str, Any]], msg: str
                             ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
            """Run the configured review (VLM Panel A when `review_models`, else the single
            independent reviewer) on one source. `val` is that candidate's validation record;
            on a NOVEL-goal vacuous pass its selectivity-probe scores are surfaced to the
            reviewer (so it doesn't false-flag the all-zero fixed battery as near-constant).
            Returns (review, review_panel)."""
            val = val or {}
            archetype_scores = val.get("archetype_scores", {})
            selectivity = (val.get("selectivity_probe")
                           if val.get("nondegeneracy_vacuous") else None)
            _emit({"stage": "reviewing", "max": n_attempts, "message": msg})
            if review_models:
                reviewers = _reviewers_from_models(list(review_models), model)
                panel = _review_metric_panel(
                    client, behavior_goal, src, archetype_scores,
                    reviewers=reviewers, keyframes=review_keyframes,
                    selectivity=selectivity)
                return panel["review"], {k: v for k, v in panel.items() if k != "review"}
            return (_review_metric(client, model, behavior_goal, src, archetype_scores,
                                   selectivity=selectivity), None)

        review_out: Optional[dict[str, Any]] = None
        review_panel: Optional[dict[str, Any]] = None
        if passed and review:
            if n_candidates and n_candidates > 1 and len(ranked_valid) > 1:
                # §best-of-N review-in-order: the OFFLINE discriminator can't predict the LLM
                # reviewer, so review the valid candidates best-first and accept the FIRST one
                # the reviewer approves — promoting it to source/validation/metric.py. A flawed
                # top candidate no longer sinks the run when a slightly less-discriminating
                # sibling passes review. Stops at the first approval (≤ N reviews).
                for k, cand in enumerate(ranked_valid):
                    review_out, review_panel = _dispatch_review(
                        cand["src"], cand["val"],
                        f"Reviewing candidate {cand['candidate'] + 1} "
                        f"({k + 1}/{len(ranked_valid)} by discrimination)…")
                    if review_out and review_out.get("approved"):
                        source, validation = cand["src"], cand["val"]
                        selected_candidate = cand["candidate"]
                        metric_path.write_text(source, encoding="utf-8")
                        break
                # none approved → review_out is the last (rejected) one; source/validation
                # stay as the discrimination-winner (already on disk); accepted stays False.
            else:
                review_out, review_panel = _dispatch_review(
                    source, validation,
                    "Passed validation — running independent review…")

        accepted = bool(passed and (review_out is None or review_out.get("approved")))

        # §review-feedback retry: a metric that PASSED validation but the reviewer VETOED is
        # otherwise a dead-end — the feedback-retry loop above fires only on a VALIDATION
        # failure. Feed the reviewer's concerns back and regenerate (bounded) so a fixable
        # objection (a missing signed-direction check, a too-soft gate) gets corrected
        # instead of failing the run. Only runs with review on + a concrete veto.
        rev_retry = 0
        while (review and passed and not accepted
               and review_out and not review_out.get("approved")
               and rev_retry < _MAX_REVIEW_RETRIES):
            rev_retry += 1
            _emit({"stage": "regenerating", "max": _MAX_REVIEW_RETRIES,
                   "message": f"Reviewer vetoed — regenerating with the review feedback "
                              f"(retry {rev_retry}/{_MAX_REVIEW_RETRIES})…"})
            user = (
                base_user
                + "\n\nThe previous metric PASSED validation but the REVIEWER REJECTED it "
                + "for these concerns:\n"
                + json.dumps(review_out.get("concerns") or [], indent=2)
                + "\nRewrite the metric to FIX every concern (keep the sharp completion gate "
                + "× min-of-saturating-channels form). Output ONLY the corrected module.")
            try:
                source = _sample_source(client, system_prompt, user, model=model)
            except Exception as e:  # noqa: BLE001 — a failed retry call ends the retries
                attempts.append({"review_retry": rev_retry,
                                 "api_error": f"{type(e).__name__}: {e}"})
                break
            selected_candidate = None        # the result is now a feedback-corrected retry,
            metric_path.write_text(source, encoding="utf-8")   # not a best-of-N candidate
            validation = validate_generated_metric(
                source, metric_path, behavior_goal=behavior_goal, robot_hint=robot_hint,
                robot_joint_names=robot_names, references=references,
                eval_reset=eval_reset, channel_catalog=catalog)
            attempts.append({"review_retry": rev_retry, "ok": validation["ok"],
                             "reasons": validation["reasons"]})
            passed = bool(validation["ok"])
            if not passed:
                break                        # the correction broke validation — stop thrashing
            review_out, review_panel = _dispatch_review(
                source, validation,
                f"Re-reviewing the corrected metric (retry {rev_retry})…")
            accepted = bool(review_out and review_out.get("approved"))

        _emit({"stage": "done", "accepted": accepted,
               "message": ("Metric accepted." if accepted
                           else "Metric rejected — see reasons.")})
        record = {
            "accepted": accepted,
            "validation_passed": passed,
            "metric_path": str(metric_path),
            "source": source,
            "validation": validation,
            "review": review_out,
            "review_panel": review_panel,
            "attempts": attempts,
            "n_candidates": n_candidates,
            "candidates": candidates,
            "selected_candidate": selected_candidate,
            "behavior_goal": behavior_goal,
            "robot_hint": robot_hint,
            "model": model,
            # steer-rights are earned later via calibration; observe-only until then.
            "calibrated": False,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            # §REFERENCE_TRAJECTORY_PLAN §7: [] / {} (not None) when no references
            # were attached, matching the "default None -> byte-identical" contract
            # at the field-VALUE level (json round-trips [] the same either way).
            "references_used": references_used,
            "reference_signatures": reference_signatures,
            # §D24 F2: the eval-reset preview (None when omitted — the
            # same "default None -> byte-identical" convention as
            # references_used/reference_signatures above, at the
            # field-VALUE level).
            "eval_reset_preview": eval_reset,
            "channel_catalog_hash": (
                catalog.catalog_hash if catalog is not None else None),
        }
        (out_dir / "meta.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8")
        return record
    finally:
        set_llm_log_dir(_prev_llm_log_dir)
