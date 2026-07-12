"""Per-stage objective-metric generation for missions.

§MISSION_METRIC_GRANULARITY (decision record, 2026-07-06): each mission
stage steers on a FRESH, trust-gated objective metric generated from the
stage's own goal text at decomposition time. The mission-level
`fitness_metric` stays as decomposition context and as the fallback for
stages whose generated metric the pipeline rejects — the existing
`steering_metric or fitness_metric` resolution is unchanged.

Mechanically this populates the Ship-38 `Stage.steering_metric` slot
with a *mission-dir-relative* path
("stage_metrics/<name>/metric.py"). Relative refs keep `mission.json`
portable and inside the 128-char validator bound;
`resolve_stage_metric_ref` anchors them at the mission dir before
`resolve_fitness_fn`'s fail-fast resolution.

Metrics live under `<mission_dir>/stage_metrics/`, NOT inside
`<mission_dir>/stages/<name>/` — the orchestrator scaffolds each
`stages/<name>/` dir with `sculpt_init`, which (correctly) refuses a
non-empty target. Generation happens at decompose time, long before
scaffolding, so anything written inside a stage dir would brick the
stage (live-caught: g1-standing-jump halted `scaffold_errored` on
exactly this).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from sculptor.mission import Mission


#: §REFERENCE_TRAJECTORY_PLAN §2.4: the reference library keys clips by a
#: bare robot slug ("g1", "go1", ...), while `robot_hint` here is the
#: adapter/task_id-shaped string used everywhere else in this module
#: (e.g. "Mjlab-Velocity-Flat-Unitree-G1"). Same substring-match idea as
#: `sculptor.eval.robot_manifest.robot_joint_names`; unknown/absent hints
#: default to "g1" (the only populated library robot today — mirrors the
#: `robot: str = "g1"` default already used throughout `sculptor.refs`).
_ROBOT_SLUGS: tuple[str, ...] = ("go2", "go1", "g1")


def _robot_slug(robot_hint: Optional[str]) -> str:
    h = str(robot_hint or "").lower()
    for slug in _ROBOT_SLUGS:
        if slug in h:
            return slug
    return "g1"


def load_stage_reference_clip(
    stage: Any, robot: str,
) -> Optional[tuple[str, dict, Optional[dict[str, Any]]]]:
    """§D24 F1 (docs/internal/REFERENCE_BUILD_LOG.md D19/D23/D24): THE ONE
    loader every consumer of `stage.reference_clip_id` MUST go through —
    D19's "every clip-shape assumption in ONE place" rule, now extended to
    goal-aligned SUB-SPANS. Loads the library clip and, when the stage
    carries a persisted `reference_span_start_s`/`_end_s` (set by
    `sculptor.decompose`'s attach/redecompose paths or this module's lazy
    backfill — see `select_reference_span`), crops it via
    `sculptor.refs.spans.crop_span` before returning it. A stage with no
    span fields gets the FULL clip back, unchanged — that is the explicit
    "no span applies" contract (see `Stage.reference_span_start_s`'s
    docstring).

    Returns `None` when `stage.reference_clip_id` is falsy (nothing to
    load). Otherwise returns `(clip_id, clip, span_meta)` where `clip` is
    the (possibly cropped) clip dict and `span_meta` is
    `{"t_start_s", "t_end_s", "confidence", "method"}` when a span was
    applied, else `None`.

    Raises on a load or crop failure — this function does NOT catch
    exceptions (unlike the rest of this module's non-fatal-by-design
    helpers), because callers have GENUINELY DIFFERENT failure policies
    for the same failure: an attached-but-unloadable clip is FATAL for
    `sculpt.py`'s RSI scaffold (§D21 Fix 2 — never silently substitute a
    different task class) but non-fatal for metric generation/calibration
    and the reference_signature.json write (§7 — degrade to no
    reference). Each caller wraps this in its own try/except.
    """
    clip_id = getattr(stage, "reference_clip_id", None)
    if not clip_id:
        return None

    from sculptor.reference import load_clip
    from sculptor.refs import library

    clip_path = library.clip_dir(robot, clip_id) / library.CLIP_FILENAME
    clip = load_clip(clip_path)

    t_start = getattr(stage, "reference_span_start_s", None)
    t_end = getattr(stage, "reference_span_end_s", None)
    span_meta: Optional[dict[str, Any]] = None
    if t_start is not None and t_end is not None:
        from sculptor.refs.spans import crop_span

        clip = crop_span(clip, t_start, t_end)
        span_meta = {
            "t_start_s": t_start,
            "t_end_s": t_end,
            "confidence": getattr(stage, "reference_span_confidence", None),
            "method": getattr(stage, "reference_span_method", None),
        }
    return clip_id, clip, span_meta


def _load_stage_reference(
    stage: Any, robot_hint: Optional[str],
) -> tuple[Optional[list[tuple[str, dict]]], Optional[str]]:
    """Load the clip attached to `stage.reference_clip_id` from the
    on-disk reference library (cropped to the stage's reference span,
    when one is persisted — via `load_stage_reference_clip`, the ONE
    loader). Returns `(references_kwarg_value, reference_load_error)` —
    `references_kwarg_value` is `[(clip_id, clip_dict)]` on success,
    `None` when the stage has no reference attached OR the clip failed to
    load (log-and-proceed WITHOUT references per §REFERENCE_TRAJECTORY_PLAN
    §7 — a missing/corrupt clip on disk must never crash the
    metric-generation pipeline). `reference_load_error` is a short string
    recorded on the stage's report entry, only set on a load failure."""
    clip_id = getattr(stage, "reference_clip_id", None)
    if not clip_id:
        return None, None
    try:
        robot = _robot_slug(robot_hint)
        loaded = load_stage_reference_clip(stage, robot)
        if loaded is None:  # pragma: no cover — clip_id truthy just above
            return None, None
        loaded_clip_id, clip, _span_meta = loaded
        return [(loaded_clip_id, clip)], None
    except Exception as e:  # noqa: BLE001 — never crash the metric pipeline
        error = f"{type(e).__name__}: {e}"
        print(
            f"[mission-metrics] stage {getattr(stage, 'name', '?')!r}: "
            f"failed to load reference clip {clip_id!r} ({error}) — "
            "proceeding WITHOUT reference grounding.",
            file=sys.stderr, flush=True)
        return None, error


def _reference_source_kind(robot: str, clip_id: str) -> str:
    """Best-effort lookup of the clip's provenance `source.kind` (e.g.
    "hf_dataset", "retarget") — the trailing segment of the `trust_tier`
    label (§REFERENCE_TRAJECTORY_PLAN §10, `"reference:<K|D>:<source>"`).
    Never raises; an unreadable/missing provenance record degrades to
    `"unknown"` rather than blocking calibration."""
    try:
        from sculptor.refs import library

        prov = library.read_provenance(robot, clip_id)
        return str((prov.get("source") or {}).get("kind") or "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def _calibrate_stage_reference(
    *, metric_path: Path, clip_id: str, clip: dict, robot: str,
) -> Optional[dict[str, Any]]:
    """§REFERENCE_TRAJECTORY_PLAN §6, wired behind the verified-cert gate.

    REFERENCE_BUILD_LOG.md "Audit findings deferred" (Tier-D spoofing):
    `calibrate_metric_against_reference` no longer accepts a caller-
    supplied tier string — it derives the effective tier itself via
    `sculptor.refs.track.verify_tierd_certificate`, so this call cannot
    manufacture steer-rights. Tier-K ladder calibration runs
    unconditionally here (grants observe when it passes); Tier-D (steer)
    is granted ONLY when a verified on-disk tracking certificate resolves
    for this exact clip. This function never CREATES a certificate (no
    GPU tracking happens here) — it only consumes one if `sculptor.refs.
    track.track_clip` already produced it.

    Non-fatal by design: any error here is caught, logged, and returns
    `None` — the stage's metric acceptance (already recorded by the
    caller) and its pre-existing trust are left exactly as they were."""
    try:
        from sculptor.eval.metric_calibration import (
            calibrate_metric_against_reference,
        )

        source_kind = _reference_source_kind(robot, clip_id)
        return calibrate_metric_against_reference(
            metric_path, clip_id, clip, robot=robot, source_kind=source_kind)
    except Exception as e:  # noqa: BLE001 — calibration must never break stage metric acceptance
        print(
            f"[mission-metrics] reference calibration for clip {clip_id!r} "
            f"failed ({type(e).__name__}: {e}) — leaving trust unchanged.",
            file=sys.stderr, flush=True)
        return None


def _compute_eval_reset_preview(
    clip: dict, *, robot: str,
) -> Optional[dict[str, Any]]:
    """§D24 F2 item 1 (docs/internal/REFERENCE_BUILD_LOG.md D17/D20a/D23):
    at cert time, compute the stage's expected EVAL-ROLLOUT START state —
    `derive_eval_reset(clip)` run through the exact SAME settle path
    `sculpt.py`'s scaffold block 2.5 uses (`sculptor.reference.
    settle_reset`, called with the identical args:
    `settle_reset(pre, joint_names=clip.get("joint_names"), robot=robot)`)
    — so these numbers and the ones the scaffold LATER writes to
    `env/eval_reset.json` agree BY CONSTRUCTION, not by coincidence
    (§D19's "one loader/one derivation path" rule, extended to this new
    consumer).

    Returns `None` when `clip` is not a low-start archetype
    (`derive_eval_reset` returns `None` — a jump/standing-start clip has
    no non-default eval-rollout start to preview, exactly like the
    scaffold writes no `eval_reset.json` for those). Otherwise returns
    `{"scalars": <reset scalars>, "settled": bool, "reason":
    Optional[str]}`:

      * settling SUCCEEDED — `scalars` are the SETTLED values,
        `settled=True`, `reason=None` (plus `delta_z_m`/`converged` for
        diagnostics).
      * settling is unavailable/disabled/raised — `scalars` are the RAW
        (unsettled) `derive_eval_reset` output, `settled=False`, and
        `reason` names why. `RS_SETTLE_RESET=0` (same escape hatch the
        scaffold honors) and any `SettleUnavailable`/other exception both
        land here.

    Never raises — this is advisory authoring/validation context, never
    a stage-blocking dependency (mirrors `settle_reset`'s own §5
    contract: settling is a REFINEMENT, never required)."""
    from sculptor.reference import derive_eval_reset

    try:
        pre = derive_eval_reset(clip)
    except Exception as e:  # noqa: BLE001 — preview is advisory, never fatal
        return {
            "scalars": None, "settled": False,
            "reason": f"derive_eval_reset failed: {type(e).__name__}: {e}",
        }
    if pre is None:
        return None

    if os.environ.get("RS_SETTLE_RESET", "1") == "0":
        return {"scalars": pre, "settled": False, "reason": "RS_SETTLE_RESET=0"}

    try:
        from sculptor.reference import settle_reset

        settle_result = settle_reset(
            pre, joint_names=clip.get("joint_names"), robot=robot)
        return {
            "scalars": settle_result["scalars"],
            "settled": True,
            "reason": None,
            "delta_z_m": settle_result["delta_z_m"],
            "converged": settle_result["converged"],
        }
    except Exception as e:  # noqa: BLE001 — settling is a refinement, never fatal
        return {
            "scalars": pre, "settled": False,
            "reason": f"{type(e).__name__}: {e}",
        }


def _backfill_stage_reference_span(
    stage: Any, robot_hint: Optional[str],
) -> Optional[str]:
    """§D24 F1 item 4c (lazy backfill): a stage with `reference_clip_id`
    set but no `reference_span_*` fields — either decomposed before span
    selection existed, or whose decompose-time selection call was
    declined/never ran — gets ONE selection attempt here. This is how an
    EXISTING mission (already decomposed) gets a span without a full
    re-decomposition.

    Mutates `stage` in place on success; persistence rides the existing
    "caller re-saves mission.json via `save_mission(mission, md)`"
    contract every `generate_stage_metrics` caller already follows (see
    this module's docstring + `docs/internal/REFERENCE_BUILD_LOG.md`
    D24) — no separate write happens here, same as every other stage
    mutation `generate_stage_metrics` makes (`steering_metric`, etc.).

    Returns `None` when no backfill was needed (no clip attached, a span
    already exists, or a prior attempt already SEMANTICALLY declined —
    see below) or the backfill succeeded; otherwise a short reason string
    (mirrors `reference_load_error`) for the caller to record on the
    stage's report entry. Never raises.

    §D24 W5 hardening (live-confirmed defect, 2026-07-12): this used to
    guard ONLY on `reference_span_start_s is not None` (the success
    marker) — a stage whose selection SEMANTICALLY declines (whole_clip /
    low_confidence / qc_reject) has all four span fields at None, so
    every subsequent `generate_stage_metrics` pass over that stage (e.g.
    the default whole-mission pass re-visiting a stage whose metric keeps
    getting rejected) re-fired a REAL `span_select` LLM call. Now checks
    `sculptor.refs.spans.is_span_declined(stage)` too, and — on a fresh
    decline — persists the `"declined:<reason>"` marker
    (`is_semantic_decline`) so the SAME semantic verdict is never
    re-attempted. An INFRA failure (llm_unavailable/parse_error/
    invalid_clip/signature_error, or a clip-load exception) persists
    NOTHING, so a transient outage is retried on the next pass."""
    if not getattr(stage, "reference_clip_id", None):
        return None
    if getattr(stage, "reference_span_start_s", None) is not None:
        return None  # already selected (or a prior attempt already ran)

    from sculptor.refs.spans import is_span_declined

    if is_span_declined(stage):
        return None  # already attempted and semantically declined

    try:
        from sculptor.reference import load_clip
        from sculptor.refs import library
        from sculptor.refs.spans import select_reference_span

        robot = _robot_slug(robot_hint)
        clip_path = (
            library.clip_dir(robot, stage.reference_clip_id)
            / library.CLIP_FILENAME)
        clip = load_clip(clip_path)
    except Exception as e:  # noqa: BLE001 — backfill is best-effort
        return f"{type(e).__name__}: {e}"

    span, reason = select_reference_span(
        clip, goal_text=stage.goal_text,
        start_pose=getattr(stage, "start_pose", None))
    if span is None:
        from sculptor.refs.spans import is_semantic_decline

        if is_semantic_decline(reason):
            stage.reference_span_method = f"declined:{reason}"
        return reason

    stage.reference_span_start_s = span["t_start_s"]
    stage.reference_span_end_s = span["t_end_s"]
    stage.reference_span_confidence = span["confidence"]
    stage.reference_span_method = span["method"]
    return None


def _unresolved_span_decline_reason(stage: Any) -> Optional[str]:
    """§H2 audit fix (docs/internal/REFERENCE_BUILD_LOG.md D23/D24/D25,
    fresh-context Opus adversarial audit of the D24/D25 batch): every
    span-selection decline used to funnel to full-clip certification —
    the EXACT D23 configuration (a sub-phase stage goal certified
    against its clip's full motion, zeroing a physically correct
    rollout) — because `load_stage_reference_clip` returns the full clip
    whenever no span is persisted, and a `declined:*` marker
    (`sculptor.refs.spans.is_span_declined`) permanently blocks retry.

    Returns the bare decline reason (e.g. `"low_confidence:0.42"`,
    `"qc_reject:crop_error:..."`) when the stage's persisted
    `reference_span_method` is a decline marker whose sub-span question
    is UNRESOLVED — i.e. every semantic decline EXCEPT `whole_clip`.
    `declined:whole_clip` means the LLM affirmatively judged the goal
    covers the clip's ENTIRE motion — full-clip certification is the
    CORRECT behavior there, so this returns `None` for it (same as "no
    decline at all"), and the caller proceeds exactly as before.

    Never raises."""
    from sculptor.refs.spans import DECLINED_METHOD_PREFIX, is_span_declined

    if not is_span_declined(stage):
        return None
    method = str(getattr(stage, "reference_span_method", "") or "")
    reason = method[len(DECLINED_METHOD_PREFIX):]
    if reason == "whole_clip":
        return None
    return reason


def resolve_stage_metric_ref(ref: str, mission_dir: Path | str) -> str:
    """Anchor a mission-dir-relative generated-metric ref at the mission
    dir. Spec-metric names ("g1_kick") and absolute paths pass through
    untouched — only a RELATIVE `*.py` ref is joined."""
    p = Path(ref)
    if p.suffix == ".py" and not p.is_absolute():
        return str(Path(mission_dir) / p)
    return ref


def generate_stage_metrics(
    mission: Mission,
    *,
    robot_hint: Optional[str] = None,
    client: Any = None,
    n_candidates: int = 1,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    only_stages: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Generate one objective metric per pending stage that has no
    `steering_metric` yet. Full trust pipeline per stage (L0 gates +
    review); an accepted metric sets `steering_metric` to the
    mission-dir-relative path, a rejected one leaves the stage on the
    mission-level fallback. Mutates `mission` in place; the caller
    re-saves. Never raises for a single stage's failure — the mission
    must stay runnable on fallbacks.

    `only_stages` (§mission-persistence increment 1): when given, only
    stages whose name is in the list are processed, and — for those
    named stages ONLY — the "skip if steering_metric already set" guard
    is BYPASSED, so a user can explicitly trigger regeneration of a
    stage that already has a metric. The "skip if already succeeded"
    guard still applies (never regenerate the metric a stage already
    won on). Stages not named in `only_stages` are left untouched (not
    even reported as "skipped" — they were never candidates).

    When `only_stages` is None (the default, whole-mission pass),
    "superseded" stages are also skipped — they are terminal and will
    never train again, so generating a metric for one wastes an LLM
    call. A user-targeted `only_stages` regenerate is still allowed to
    hit a superseded stage (e.g. to refresh what a report shows), which
    is why that check is IN the default branch, not a blanket guard.

    Returns `{"generated": [...], "rejected": [...], "skipped": [...]}`
    where each entry is `{stage, reason?}`.
    """
    from sculptor.eval import generate_objective_metric

    if mission.mission_dir is None:
        raise RuntimeError(
            "mission.mission_dir is None — save_mission before "
            "generate_stage_metrics so stage metric dirs resolve.")
    mission_dir = Path(mission.mission_dir)

    def _emit(ev: dict[str, Any]) -> None:
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:  # noqa: BLE001 — progress is advisory
                pass

    report: dict[str, list[dict[str, Any]]] = {
        "generated": [], "rejected": [], "skipped": [],
    }
    only_set = set(only_stages) if only_stages is not None else None
    for stage in mission.stages:
        if only_set is not None:
            if stage.name not in only_set:
                continue
        else:
            if getattr(stage, "steering_metric", None):
                report["skipped"].append(
                    {"stage": stage.name, "reason": "steering_metric already set"})
                continue
            if getattr(stage, "status", "pending") == "superseded":
                report["skipped"].append(
                    {"stage": stage.name, "reason": "stage superseded"})
                continue
        if getattr(stage, "status", "pending") == "succeeded":
            report["skipped"].append(
                {"stage": stage.name, "reason": "stage already succeeded"})
            continue
        out_dir = mission_dir / "stage_metrics" / stage.name
        _emit({
            "type": "stage_metric_gen_started",
            "stage": stage.name,
            "goal_text": stage.goal_text[:200],
        })
        # §REFERENCE_TRAJECTORY_PLAN §7: a stage with an attached library
        # reference gets it loaded + threaded through generation for
        # kinematic grounding + reference-anchored validation. No
        # auto-retrieval here (out of scope — human attaches via UI); a
        # missing/corrupt clip degrades to no-reference rather than
        # failing the stage.
        #
        # §D24 F1 item 4c: lazy span backfill runs FIRST — a stage
        # decomposed before span selection existed (e.g. the live
        # g1-standing mission) has `reference_clip_id` set but no span
        # fields; this is the one-time selection that fixes it, without
        # a full re-decomposition. Must run BEFORE `_load_stage_reference`
        # so the freshly-persisted span is what gets loaded+cropped for
        # THIS generation pass.
        had_span_before_backfill = (
            getattr(stage, "reference_span_start_s", None) is not None)
        span_backfill_reason = _backfill_stage_reference_span(stage, robot_hint)
        backfilled_now = (
            not had_span_before_backfill
            and span_backfill_reason is None
            and getattr(stage, "reference_span_start_s", None) is not None)
        if backfilled_now:
            _emit({
                "type": "stage_reference_span_backfilled",
                "stage": stage.name,
                "clip_id": stage.reference_clip_id,
                "t_start_s": stage.reference_span_start_s,
                "t_end_s": stage.reference_span_end_s,
            })
        elif span_backfill_reason is not None:
            _emit({
                "type": "stage_reference_span_backfill_skipped",
                "stage": stage.name,
                "clip_id": stage.reference_clip_id,
                "reason": span_backfill_reason,
            })

        # §H2 audit fix (docs/internal/REFERENCE_BUILD_LOG.md D23/D24/D25):
        # an UNRESOLVED span decline (anything but `declined:whole_clip`)
        # means the sub-span question was never answered — certifying
        # against the full clip here would reproduce the exact D23 class
        # (a sub-phase goal certified against the wrong-scope clip,
        # zeroing a physically correct rollout). Skip generation entirely
        # for this stage; it falls back to the mission-level
        # `fitness_metric` (§MISSION_METRIC_GRANULARITY's existing
        # `steering_metric or fitness_metric` resolution in sculpt.py) —
        # criterion + start-state gate enforcement are UNAFFECTED (they
        # never route through `steering_metric`). A user can retry via
        # the per-stage regenerate endpoint, which clears the marker
        # (see `reward-sculptor-ui/backend/routes/missions.py`
        # `regenerate_stage_metric`).
        unresolved_decline_reason = _unresolved_span_decline_reason(stage)
        if unresolved_decline_reason is not None:
            reject_reason = (
                "reference span selection declined "
                f"({unresolved_decline_reason}); refusing full-clip "
                "certification for a possibly sub-phase goal (D23 "
                "class); the stage will use the mission-level fallback "
                "— re-run per-stage metric regeneration to retry span "
                "selection"
            )
            reject_entry: dict[str, Any] = {
                "stage": stage.name, "reason": reject_reason}
            if span_backfill_reason:
                reject_entry["reference_span_backfill_reason"] = span_backfill_reason
            report["rejected"].append(reject_entry)
            _emit({
                "type": "stage_metric_gen_rejected",
                "stage": stage.name,
                "reason": reject_reason,
            })
            continue

        references, reference_load_error = _load_stage_reference(
            stage, robot_hint)

        # §M3 audit fix (docs/internal/REFERENCE_BUILD_LOG.md, fresh-
        # context Opus adversarial audit): an ATTACHED clip
        # (`reference_clip_id` set) that fails to load/crop must NOT
        # silently downgrade to an UNGATED, no-reference metric
        # acceptance — `_load_stage_reference` catches every load/crop
        # exception, and validation would otherwise run with zero
        # reference gates, accepting a metric the author never intended
        # to ship without its exemplar. A stage with NO
        # `reference_clip_id` at all is unaffected — no-reference
        # validation is legitimate for it (this only fires when a clip
        # WAS attached and the load genuinely failed, i.e.
        # `reference_load_error` is set).
        if getattr(stage, "reference_clip_id", None) and reference_load_error is not None:
            reject_reason = (
                "attached reference failed to load "
                f"({reference_load_error}); refusing certification "
                "without its intended exemplar"
            )
            reject_entry = {
                "stage": stage.name, "reason": reject_reason,
                "reference_load_error": reference_load_error,
            }
            if span_backfill_reason:
                reject_entry["reference_span_backfill_reason"] = span_backfill_reason
            report["rejected"].append(reject_entry)
            _emit({
                "type": "stage_metric_gen_rejected",
                "stage": stage.name,
                "reason": reject_reason,
            })
            continue

        # §D24 F2 item 1: compute the stage's certified eval-reset preview
        # from the SAME (possibly cropped) clip just loaded for
        # `references` — persisted next to `meta.json` and threaded into
        # generation (authoring context) + validation (the
        # `reference_settled_start` gate). None whenever there is no
        # reference (or it isn't a low-start archetype) — never fatal.
        eval_reset_preview: Optional[dict[str, Any]] = None
        if references:
            ref_clip_id_for_reset, ref_clip_for_reset = references[0]
            eval_reset_robot = _robot_slug(robot_hint)
            try:
                eval_reset_preview = _compute_eval_reset_preview(
                    ref_clip_for_reset, robot=eval_reset_robot)
            except Exception as e:  # noqa: BLE001 — preview is advisory
                print(
                    f"[mission-metrics] stage {stage.name!r}: eval-reset "
                    f"preview computation failed ({type(e).__name__}: "
                    f"{e}) — proceeding without it.",
                    file=sys.stderr, flush=True)
                eval_reset_preview = None
            if eval_reset_preview is not None:
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "eval_reset_preview.json").write_text(
                        json.dumps(eval_reset_preview, indent=2, default=str),
                        encoding="utf-8")
                except Exception as e:  # noqa: BLE001 — persistence is advisory
                    print(
                        f"[mission-metrics] stage {stage.name!r}: failed "
                        f"to write eval_reset_preview.json "
                        f"({type(e).__name__}: {e}).",
                        file=sys.stderr, flush=True)

            # §D24 F2 (D deliverable): re-ground the stage's blind-authored
            # success_criterion in the CROPPED clip's real signature — the
            # SAME lazy mechanism that fixes an EXISTING mission (e.g.
            # g1-standing's torso_righting) without a full re-decomposition.
            # Gated on `backfilled_now` (the span was JUST discovered by
            # THIS call) so this real LLM call fires exactly once per span
            # discovery, never on every subsequent pass over a stage whose
            # metric keeps getting rejected (same repeat-call discipline
            # the span-backfill declined-marker fix above enforces).
            # Stages whose span already existed from decompose time were
            # already re-grounded there (`_select_and_attach_span`).
            if backfilled_now:
                try:
                    from sculptor.decompose import ground_stage_criterion

                    ground_stage_criterion(stage, ref_clip_for_reset)
                except Exception as e:  # noqa: BLE001 — never fatal
                    print(
                        f"[mission-metrics] stage {stage.name!r}: "
                        f"criterion re-grounding crashed "
                        f"({type(e).__name__}: {e}) — keeping the "
                        f"original criterion.", file=sys.stderr, flush=True)

        try:
            rec = generate_objective_metric(
                stage.goal_text, out_dir,
                robot_hint=robot_hint, client=client,
                n_candidates=n_candidates,
                on_event=on_event,
                references=references,
                eval_reset=eval_reset_preview,
            )
        except Exception as e:  # noqa: BLE001 — stage falls back, mission runs
            print(
                f"[mission-metrics] stage {stage.name!r}: generation "
                f"crashed ({type(e).__name__}: {e}) — stage falls back to "
                f"the mission-level metric.", file=sys.stderr, flush=True)
            reject_entry = {"stage": stage.name,
                            "reason": f"{type(e).__name__}: {e}"}
            if reference_load_error:
                reject_entry["reference_load_error"] = reference_load_error
            if span_backfill_reason:
                reject_entry["reference_span_backfill_reason"] = span_backfill_reason
            report["rejected"].append(reject_entry)
            _emit({
                "type": "stage_metric_gen_failed",
                "stage": stage.name,
                "reason": f"{type(e).__name__}: {e}",
            })
            continue
        if rec.get("accepted"):
            rel = f"stage_metrics/{stage.name}/metric.py"
            stage.steering_metric = rel
            gen_entry = {"stage": stage.name, "ref": rel}
            if reference_load_error:
                gen_entry["reference_load_error"] = reference_load_error
            if span_backfill_reason:
                gen_entry["reference_span_backfill_reason"] = span_backfill_reason
            report["generated"].append(gen_entry)
            _emit({
                "type": "stage_metric_gen_accepted",
                "stage": stage.name,
                "ref": rel,
            })
            # §REFERENCE_TRAJECTORY_PLAN §6, wired behind the verified-cert
            # gate (audit-finding close): a stage whose metric was just
            # certified/accepted against an attached reference also earns
            # reference-derived calibration trust here. Byte-level no-op
            # for stages without a reference (`references` is None/empty).
            if references:
                ref_clip_id, ref_clip = references[0]
                ref_robot = _robot_slug(robot_hint)
                cal = _calibrate_stage_reference(
                    metric_path=out_dir / "metric.py", clip_id=ref_clip_id,
                    clip=ref_clip, robot=ref_robot)
                if cal is not None:
                    cal_summary = {
                        "clip_id": ref_clip_id,
                        "robot": ref_robot,
                        "trust_tier": cal.get("trust_tier"),
                        "rights": cal.get("rights"),
                        "ok": cal.get("ok"),
                        "cert_verified": cal.get("cert_verified"),
                        "cert_reason": cal.get("cert_reason"),
                    }
                    gen_entry["reference_calibration"] = cal_summary
                    _emit({
                        "type": "stage_metric_reference_calibrated",
                        "stage": stage.name,
                        **cal_summary,
                    })
        else:
            # generate_objective_metric never raises on a bad candidate —
            # rejection reasons live in the record it wrote to meta.json.
            reasons = rec.get("validation") or {}
            reason = "; ".join(
                (reasons.get("reasons") or ["review rejected"])[:3])
            reject_entry = {"stage": stage.name, "reason": reason}
            if reference_load_error:
                reject_entry["reference_load_error"] = reference_load_error
            if span_backfill_reason:
                reject_entry["reference_span_backfill_reason"] = span_backfill_reason
            report["rejected"].append(reject_entry)
            _emit({
                "type": "stage_metric_gen_rejected",
                "stage": stage.name,
                "reason": reason,
            })
    return report
