"""Benchmark task suite (§Ship 26 / E1).

Four tasks spanning difficulty, each pairing an mjlab task_id + a
natural-language behavior goal (what the sculptor pipeline gets) with a
HAND-AUTHORED spec metric (what the evaluation believes — see
spec_metrics.py). The E2 harness iterates this table; E3 baselines and
E4 ablation conditions run the same tasks so comparisons are paired.

Why these four:
  * cartpole_balance — cheap sanity task (256 envs, minutes/run): the
    harness self-test target and the high-seed-count task for variance
    estimation.
  * g1_floss / g1_kick — the two behaviors Sam has run real missions on
    (g1-flossing, g1-kick-v2/3 projects), so spec metrics can be
    validated against existing labeled-by-outcome recordings.
  * go1_trot — quadruped locomotion; gait rather than spin because the
    rollout does not persist yaw (projected gravity is yaw-invariant).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BENCHMARK_MANIFEST_SCHEMA_VERSION = 1
EVALUATION_TIERS = frozenset({
    "compile_only", "rollout_artifact", "heldout_solution",
})
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_MANIFEST_FIELDS = {
    "schema_version", "suite_id", "suite_version", "benchmarks", "notes",
}
_BENCHMARK_FIELDS = {
    "name", "task_id", "behavior_goal", "spec_metric", "adapter",
    "adapter_config", "robot_id", "embodiment_family", "task_family",
    "required_capabilities", "evaluation_tier", "campaign_ready",
    "known_limitations", "spec_authority", "spec_audit_certificate", "notes",
}


class BenchmarkManifestError(ValueError):
    """An external benchmark manifest is malformed or scientifically unsafe."""


@dataclass(frozen=True)
class BenchmarkTask:
    """One benchmark entry. `behavior_goal` is the EXACT NL string fed
    to the pipeline (decompose/diagnose see only this); `spec_metric`
    names the ground-truth function in spec_metrics.py."""

    name: str
    task_id: str
    behavior_goal: str
    spec_metric: str | None
    adapter: str = "mjlab"
    #: Default adapter-config overrides for eval runs (envs sized for
    #: throughput on the eval GPU; the harness may override).
    adapter_config: dict[str, Any] = field(default_factory=dict)
    #: Embodiment/capability metadata drives suite coverage without brittle
    #: robot-name conditionals. These labels are descriptive; adapter support
    #: remains the executable authority.
    robot_id: str = "unspecified"
    embodiment_family: str = "unspecified"
    task_family: str = "unspecified"
    required_capabilities: tuple[str, ...] = ()
    #: compile_only tasks document a frontier but cannot enter a campaign;
    #: rollout_artifact tasks have an objective artifact metric;
    #: heldout_solution additionally has frozen unseen evaluation evidence.
    evaluation_tier: str = "rollout_artifact"
    campaign_ready: bool = True
    #: Built-ins predate the certificate workflow and remain explicitly
    #: provisional until audited. New external campaign entries require A4.
    spec_authority: str = "legacy_provisional"
    spec_audit_certificate_sha256: str | None = None
    known_limitations: tuple[str, ...] = ()
    #: Why this task / what the spec measures — surfaced in reports.
    notes: str = ""


BENCHMARKS: dict[str, BenchmarkTask] = {
    b.name: b
    for b in (
        BenchmarkTask(
            name="cartpole_balance",
            task_id="Mjlab-Cartpole-Balance",
            behavior_goal="balance the pole upright and keep the cart centered",
            spec_metric="cartpole_balance",
            adapter_config={"num_envs": 256},
            robot_id="cartpole",
            embodiment_family="underactuated_system",
            task_family="control_sanity",
            required_capabilities=("balance",),
            notes=(
                "Sanity task: spec = normalized mean episode length. "
                "Cheap enough for high seed counts."
            ),
        ),
        BenchmarkTask(
            name="g1_floss",
            task_id="Mjlab-Velocity-Flat-Unitree-G1",
            behavior_goal=(
                "perform a continuous flossing dance: swing the hips "
                "side to side while the arms swing in opposition"
            ),
            spec_metric="g1_floss",
            adapter_config={"num_envs": 4096},
            robot_id="unitree_g1",
            embodiment_family="humanoid",
            task_family="expressive_whole_body_motion",
            required_capabilities=("whole_body_control", "balance"),
            notes=(
                "Spec = joint-space periodicity (dominant-period power "
                "ratio over the most-moving joints) x uprightness."
            ),
        ),
        BenchmarkTask(
            name="g1_kick",
            task_id="Mjlab-Velocity-Flat-Unitree-G1",
            behavior_goal=(
                "repeatedly kick forward with one leg while keeping "
                "balance on the other"
            ),
            spec_metric="g1_kick",
            adapter_config={"num_envs": 4096},
            robot_id="unitree_g1",
            embodiment_family="humanoid",
            task_family="dynamic_single_support",
            required_capabilities=("legged_locomotion", "balance"),
            notes=(
                "Spec = leg-speed burst intensity x burst-vs-hum ratio "
                "x uprightness."
            ),
        ),
        BenchmarkTask(
            name="go1_trot",
            task_id="Mjlab-Velocity-Flat-Unitree-Go1",
            behavior_goal="trot forward in a straight line at a steady pace",
            spec_metric="go1_trot",
            adapter_config={"num_envs": 4096},
            robot_id="unitree_go1",
            embodiment_family="quadruped",
            task_family="locomotion",
            required_capabilities=("legged_locomotion",),
            notes=(
                "Spec = saturating horizontal speed x straightness x "
                "uprightness. Gait, not spin: yaw is not persisted."
            ),
        ),
    )
}


def _text(raw: dict[str, Any], field_name: str, context: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkManifestError(
            f"{context}.{field_name} must be a non-empty string"
        )
    return value.strip()


def _reject_unknown(
    raw: dict[str, Any], allowed: set[str], context: str,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise BenchmarkManifestError(
            f"{context} has unknown fields {unknown}; refusing possible typos"
        )


def _string_list(
    raw: dict[str, Any], field_name: str, context: str, *, required: bool = False,
) -> tuple[str, ...]:
    value = raw.get(field_name, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise BenchmarkManifestError(
            f"{context}.{field_name} must be a list of non-empty strings"
        )
    result = tuple(item.strip() for item in value)
    if required and not result:
        raise BenchmarkManifestError(f"{context}.{field_name} cannot be empty")
    if len(set(result)) != len(result):
        raise BenchmarkManifestError(f"{context}.{field_name} has duplicates")
    return result


def load_benchmark_manifest(path: Path | str) -> dict[str, BenchmarkTask]:
    """Load a strict external suite fragment.

    External manifests add tasks but never override built-ins. A compile-only
    arm/gripper frontier is useful and visible, but cannot accidentally enter
    a GPU campaign before objective rollout telemetry and a spec exist.
    """
    path = Path(path).expanduser().resolve()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkManifestError(f"cannot read benchmark manifest {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise BenchmarkManifestError("benchmark manifest must be a JSON object")
    _reject_unknown(doc, _MANIFEST_FIELDS, "manifest")
    if doc.get("schema_version") != BENCHMARK_MANIFEST_SCHEMA_VERSION:
        raise BenchmarkManifestError(
            f"benchmark manifest schema_version must be "
            f"{BENCHMARK_MANIFEST_SCHEMA_VERSION}"
        )
    _text(doc, "suite_id", "manifest")
    _text(doc, "suite_version", "manifest")
    entries = doc.get("benchmarks")
    if not isinstance(entries, list) or not entries:
        raise BenchmarkManifestError("manifest.benchmarks must be a non-empty list")

    from sculptor.eval.spec_metrics import spec_metric_names

    known_metrics = set(spec_metric_names())
    loaded: dict[str, BenchmarkTask] = {}
    for index, value in enumerate(entries):
        context = f"benchmarks[{index}]"
        if not isinstance(value, dict):
            raise BenchmarkManifestError(f"{context} must be an object")
        _reject_unknown(value, _BENCHMARK_FIELDS, context)
        name = _text(value, "name", context)
        if not _NAME_RE.fullmatch(name):
            raise BenchmarkManifestError(
                f"{context}.name must match {_NAME_RE.pattern!r}"
            )
        if name in loaded:
            raise BenchmarkManifestError(f"duplicate benchmark name {name!r}")
        tier = _text(value, "evaluation_tier", context)
        if tier not in EVALUATION_TIERS:
            raise BenchmarkManifestError(
                f"{context}.evaluation_tier must be one of {sorted(EVALUATION_TIERS)}"
            )
        campaign_ready = value.get("campaign_ready")
        if not isinstance(campaign_ready, bool):
            raise BenchmarkManifestError(f"{context}.campaign_ready must be boolean")
        spec_raw = value.get("spec_metric")
        if spec_raw is not None and (
            not isinstance(spec_raw, str) or not spec_raw.strip()
        ):
            raise BenchmarkManifestError(
                f"{context}.spec_metric must be null or a non-empty string"
            )
        spec_metric = spec_raw.strip() if isinstance(spec_raw, str) else None
        if campaign_ready:
            if tier == "compile_only":
                raise BenchmarkManifestError(
                    f"{context} is compile_only and cannot be campaign_ready"
                )
            if spec_metric not in known_metrics:
                raise BenchmarkManifestError(
                    f"{context}.spec_metric {spec_metric!r} is not registered; "
                    f"known metrics: {sorted(known_metrics)}"
                )
            authority = _text(value, "spec_authority", context)
            if authority != "A4_reporting":
                raise BenchmarkManifestError(
                    f"{context}: campaign-ready external benchmarks require "
                    "spec_authority='A4_reporting'"
                )
            certificate_raw = value.get("spec_audit_certificate")
            if not isinstance(certificate_raw, str) or not certificate_raw.strip():
                raise BenchmarkManifestError(
                    f"{context}.spec_audit_certificate is required for a "
                    "campaign-ready external benchmark"
                )
            certificate_path = Path(certificate_raw)
            if not certificate_path.is_absolute():
                certificate_path = path.parent / certificate_path
            from sculptor.eval.spec_audit import (
                SpecAuditError,
                load_and_verify_spec_certificate,
            )

            try:
                certificate = load_and_verify_spec_certificate(certificate_path)
            except SpecAuditError as exc:
                raise BenchmarkManifestError(
                    f"{context}.spec_audit_certificate is invalid: {exc}"
                ) from exc
            if (
                not certificate.get("passed")
                or certificate.get("authority_decision") != "A4_reporting"
                or certificate.get("spec_name") != spec_metric
            ):
                raise BenchmarkManifestError(
                    f"{context}.spec_audit_certificate does not grant "
                    f"A4_reporting for {spec_metric!r}"
                )
            certificate_hash = str(certificate["certificate_sha256"])
        elif tier == "compile_only" and spec_metric is not None:
            raise BenchmarkManifestError(
                f"{context}: compile_only tasks must use spec_metric=null so a "
                "placeholder cannot be mistaken for authority"
            )
        else:
            authority = _text(value, "spec_authority", context)
            certificate_hash = None
            if authority != "A0_rejected":
                raise BenchmarkManifestError(
                    f"{context}: non-ready benchmarks must use "
                    "spec_authority='A0_rejected'"
                )
            if value.get("spec_audit_certificate") is not None:
                raise BenchmarkManifestError(
                    f"{context}: non-ready benchmarks cannot cite an active "
                    "spec_audit_certificate"
                )
        adapter_config = value.get("adapter_config", {})
        if not isinstance(adapter_config, dict):
            raise BenchmarkManifestError(f"{context}.adapter_config must be an object")
        loaded[name] = BenchmarkTask(
            name=name,
            task_id=_text(value, "task_id", context),
            behavior_goal=_text(value, "behavior_goal", context),
            spec_metric=spec_metric,
            adapter=_text(value, "adapter", context),
            adapter_config=dict(adapter_config),
            robot_id=_text(value, "robot_id", context),
            embodiment_family=_text(value, "embodiment_family", context),
            task_family=_text(value, "task_family", context),
            required_capabilities=_string_list(
                value, "required_capabilities", context, required=True,
            ),
            evaluation_tier=tier,
            campaign_ready=campaign_ready,
            spec_authority=authority,
            spec_audit_certificate_sha256=certificate_hash,
            known_limitations=_string_list(
                value, "known_limitations", context,
                required=not campaign_ready,
            ),
            notes=str(value.get("notes") or ""),
        )
    return loaded


def benchmark_registry(
    manifest_paths: tuple[Path | str, ...] | list[Path | str] = (),
) -> dict[str, BenchmarkTask]:
    """Built-ins plus strict, non-overriding external manifest entries."""
    registry = dict(BENCHMARKS)
    for path in manifest_paths:
        for name, benchmark in load_benchmark_manifest(path).items():
            if name in registry:
                raise BenchmarkManifestError(
                    f"benchmark {name!r} from {Path(path)} conflicts with an "
                    "existing definition; benchmark overrides are forbidden"
                )
            registry[name] = benchmark
    return registry


def get_benchmark(
    name: str, registry: dict[str, BenchmarkTask] | None = None,
) -> BenchmarkTask:
    source = BENCHMARKS if registry is None else registry
    try:
        return source[name]
    except KeyError:
        raise KeyError(
            f"unknown benchmark {name!r}; known: {sorted(source)}"
        ) from None
