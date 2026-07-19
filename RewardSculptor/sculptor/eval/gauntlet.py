"""Blinded human-anchor studies for evaluator and metric validation.

The gauntlet is intentionally evaluator-agnostic and embodiment-agnostic.
It turns a private, labeled set of rollout videos into counterbalanced public
study packets, then joins human responses back to the private key for a
pre-specified analysis.  Reward code, run condition, iteration, source path,
metric score, and competence labels never enter the public packet.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from sculptor.eval.stats import stratified_bootstrap_ci
from sculptor.run_context import write_json_atomic


GAUNTLET_SCHEMA_VERSION = 1
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
_CHOICES = {"A", "B", "tie", "abstain"}


class GauntletError(ValueError):
    """A study design, packet, or response failed a hard validity check."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _opaque_id(*parts: Any, length: int = 20) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def _require_text(obj: Mapping[str, Any], field: str, context: str) -> str:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GauntletError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise GauntletError(f"{field} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GauntletError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise GauntletError(f"{field} must be finite")
    return number


def _load_source_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GauntletError(f"cannot read source manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise GauntletError("source manifest must be a JSON object")
    if manifest.get("schema_version") != GAUNTLET_SCHEMA_VERSION:
        raise GauntletError(
            f"source manifest schema_version must be {GAUNTLET_SCHEMA_VERSION}"
        )
    _require_text(manifest, "study_id", "manifest")
    _require_text(manifest, "rubric", "manifest")
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list) or len(raw_items) < 2:
        raise GauntletError("manifest.items must contain at least two items")

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_artifact_hashes: dict[str, str] = {}
    for index, raw in enumerate(raw_items):
        context = f"items[{index}]"
        if not isinstance(raw, dict):
            raise GauntletError(f"{context} must be an object")
        item = dict(raw)
        for field in (
            "item_id", "comparison_group", "task_id", "task_prompt",
            "robot_id", "embodiment_family", "motion_family",
            "behavior_class", "artifact_path",
        ):
            item[field] = _require_text(item, field, context)
        if item["item_id"] in seen_ids:
            raise GauntletError(f"duplicate item_id {item['item_id']!r}")
        seen_ids.add(item["item_id"])
        item["competence_rank"] = _finite_number(
            item.get("competence_rank"), f"{context}.competence_rank",
        )
        if item.get("evaluator_score") is not None:
            item["evaluator_score"] = _finite_number(
                item["evaluator_score"], f"{context}.evaluator_score",
            )

        artifact = Path(item["artifact_path"])
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        artifact = artifact.resolve()
        if not artifact.is_file():
            raise GauntletError(f"{context}.artifact_path does not exist: {artifact}")
        if artifact.suffix.lower() not in _VIDEO_EXTENSIONS:
            raise GauntletError(
                f"{context}.artifact_path must be a rollout video "
                f"({', '.join(sorted(_VIDEO_EXTENSIONS))})"
            )
        artifact_hash = _sha256_file(artifact)
        if artifact_hash in seen_artifact_hashes:
            raise GauntletError(
                f"items {seen_artifact_hashes[artifact_hash]!r} and "
                f"{item['item_id']!r} contain byte-identical videos; "
                "duplicate evidence would create pseudo-replication"
            )
        seen_artifact_hashes[artifact_hash] = item["item_id"]
        item["artifact_path"] = str(artifact)
        item["artifact_sha256"] = artifact_hash
        items.append(item)

    if any(item.get("evaluator_score") is not None for item in items):
        evaluator = manifest.get("evaluator")
        if not isinstance(evaluator, dict):
            raise GauntletError(
                "manifest.evaluator is required when evaluator_score is present"
            )
        for field in ("evaluator_id", "evaluator_version", "score_semantics"):
            _require_text(evaluator, field, "manifest.evaluator")
        if not isinstance(evaluator.get("higher_is_better"), bool):
            raise GauntletError(
                "manifest.evaluator.higher_is_better must be boolean"
            )

    # A comparison group defines one coherent question.  Mixing prompts or
    # task ids within it would turn the expected preference into nonsense.
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_group[item["comparison_group"]].append(item)
    for group, grouped in by_group.items():
        for field in ("task_id", "task_prompt"):
            values = {item[field] for item in grouped}
            if len(values) != 1:
                raise GauntletError(
                    f"comparison_group {group!r} mixes {field}: {sorted(values)}"
                )
        if len({item["competence_rank"] for item in grouped}) < 2:
            raise GauntletError(
                f"comparison_group {group!r} needs at least two competence ranks"
            )
    return manifest, items


def _balanced_pairs(
    items: Sequence[Mapping[str, Any]], *, seed: int, max_pairs_per_group: int,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Choose unequal-rank pairs with balanced behavior-class coverage."""
    if max_pairs_per_group < 1:
        raise GauntletError("max_pairs_per_group must be >= 1")
    rng = random.Random(seed)
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        by_group[str(item["comparison_group"])].append(item)

    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for group in sorted(by_group):
        candidates: dict[
            tuple[str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]
        ] = defaultdict(list)
        grouped = sorted(by_group[group], key=lambda item: str(item["item_id"]))
        for i, left in enumerate(grouped):
            for right in grouped[i + 1:]:
                if left["competence_rank"] == right["competence_rank"]:
                    continue
                stratum = tuple(sorted((
                    str(left["behavior_class"]), str(right["behavior_class"]),
                )))
                candidates[stratum].append((left, right))
        for pairs in candidates.values():
            rng.shuffle(pairs)

        # Round-robin sampling prevents a large common class from drowning out
        # rare exploit classes.  The seed fixes both within-stratum choice and
        # the stratum rotation.
        strata = sorted(candidates)
        rng.shuffle(strata)
        chosen: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        while strata and len(chosen) < max_pairs_per_group:
            remaining: list[tuple[str, str]] = []
            for stratum in strata:
                if candidates[stratum] and len(chosen) < max_pairs_per_group:
                    chosen.append(candidates[stratum].pop())
                if candidates[stratum]:
                    remaining.append(stratum)
            strata = remaining
        selected.extend(chosen)
    if not selected:
        raise GauntletError("no unequal-competence comparisons could be built")
    return selected


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - fall back to PATH
        found = shutil.which("ffmpeg")
        if found:
            return found
    raise GauntletError(
        "ffmpeg is required to strip identifying metadata from blind-study videos"
    )


def _sanitize_video(source: Path, destination: Path) -> None:
    """Remux a video while removing container metadata and chapters."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_executable(), "-v", "error", "-nostdin", "-y",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-map_metadata", "-1", "-map_chapters", "-1", "-c:v", "copy",
        str(destination),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GauntletError(f"failed to sanitize video {source}: {exc}") from exc
    if result.returncode != 0 or not destination.is_file():
        detail = (result.stderr or result.stdout or "unknown ffmpeg error").strip()
        raise GauntletError(f"failed to sanitize video {source}: {detail[-500:]}")
    # Do not leak the source's filesystem timestamps through distributed files.
    try:
        os.utime(destination, (946684800, 946684800))  # 2000-01-01 UTC
    except OSError:
        pass


def _key_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    key = dict(payload)
    key["key_sha256"] = _sha256_json(key)
    return key


def load_and_verify_study_key(path: Path) -> dict[str, Any]:
    try:
        key = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GauntletError(f"cannot read study key {path}: {exc}") from exc
    if not isinstance(key, dict):
        raise GauntletError("study key must be a JSON object")
    if key.get("schema_version") != GAUNTLET_SCHEMA_VERSION:
        raise GauntletError("unsupported study key schema_version")
    stored = key.get("key_sha256")
    unhashed = {k: v for k, v in key.items() if k != "key_sha256"}
    if stored != _sha256_json(unhashed):
        raise GauntletError("study key hash mismatch; the private key was altered")
    return key


def build_blind_study(
    source_manifest: Path,
    out_dir: Path,
    *,
    seed: int = 0,
    forms: int = 2,
    max_pairs_per_group: int = 50,
    reliability_repeats: int = 0,
    evaluator_tie_band: float = 0.0,
) -> dict[str, Any]:
    """Create blinded, counterbalanced packets and a separate private key."""
    if forms not in (1, 2):
        raise GauntletError("forms must be 1 or 2 (two gives counterbalancing)")
    if reliability_repeats < 0:
        raise GauntletError("reliability_repeats must be >= 0")
    tie_band = _finite_number(evaluator_tie_band, "evaluator_tie_band")
    if tie_band < 0:
        raise GauntletError("evaluator_tie_band must be >= 0")

    source_manifest = Path(source_manifest).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise GauntletError(
            f"output directory is not empty: {out_dir}; use a fresh directory "
            "so randomization cannot be silently replaced"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, items = _load_source_manifest(source_manifest)
    study_id = str(manifest["study_id"])
    rubric = str(manifest["rubric"])
    rng = random.Random(seed)
    base_pairs = _balanced_pairs(
        items, seed=seed, max_pairs_per_group=max_pairs_per_group,
    )
    if reliability_repeats > len(base_pairs):
        raise GauntletError(
            "reliability_repeats cannot exceed the number of primary pairs"
        )
    repeated_indices = set(rng.sample(
        range(len(base_pairs)), reliability_repeats,
    ))
    item_by_id = {str(item["item_id"]): item for item in items}

    private_pairs: dict[str, Any] = {}
    pair_order: list[str] = []
    for index, (left, right) in enumerate(base_pairs):
        ids = sorted((str(left["item_id"]), str(right["item_id"])))
        base_id = "p_" + _opaque_id(study_id, "pair", *ids)
        expected = (
            str(left["item_id"])
            if float(left["competence_rank"]) > float(right["competence_rank"])
            else str(right["item_id"])
        )
        private_pairs[base_id] = {
            "pair_id": base_id,
            "base_pair_id": base_id,
            "presentation_kind": "primary",
            "item_ids": ids,
            "expected_item_id": expected,
            "comparison_group": left["comparison_group"],
            "task_id": left["task_id"],
            "task_prompt": left["task_prompt"],
            "presentations": {},
        }
        pair_order.append(base_id)
        if index in repeated_indices:
            repeat_id = "r_" + _opaque_id(study_id, "repeat", *ids, index)
            private_pairs[repeat_id] = {
                **{k: v for k, v in private_pairs[base_id].items()
                   if k not in {"pair_id", "presentation_kind", "presentations"}},
                "pair_id": repeat_id,
                "presentation_kind": "reliability_repeat",
                "presentations": {},
            }
            pair_order.append(repeat_id)

    form_names = [chr(ord("A") + index) for index in range(forms)]
    public_packets: dict[str, dict[str, Any]] = {}
    public_assets: dict[str, dict[str, Any]] = {}
    for form_index, form_name in enumerate(form_names):
        # Every rater sees a randomized order. Form B exactly reverses A's
        # left/right assignment, enabling a clean position-bias estimate.
        form_rng = random.Random(seed + 104729)
        shuffled_ids = list(pair_order)
        form_rng.shuffle(shuffled_ids)
        public_pairs: list[dict[str, Any]] = []
        asset_for_item: dict[str, str] = {}
        for pair_id in shuffled_ids:
            pair = private_pairs[pair_id]
            item_1, item_2 = pair["item_ids"]
            random_first = bool(form_rng.getrandbits(1))
            a_item, b_item = (
                (item_1, item_2) if random_first else (item_2, item_1)
            )
            if form_index == 1:
                a_item, b_item = b_item, a_item
            pair["presentations"][form_name] = {
                "a_item_id": a_item, "b_item_id": b_item,
            }
            for item_id in (a_item, b_item):
                if item_id not in asset_for_item:
                    ext = Path(item_by_id[item_id]["artifact_path"]).suffix.lower()
                    asset_name = _opaque_id(
                        study_id, form_name, "asset", item_id,
                    ) + ext
                    relative = Path("assets") / f"form_{form_name}" / asset_name
                    _sanitize_video(
                        Path(item_by_id[item_id]["artifact_path"]),
                        out_dir / relative,
                    )
                    asset_for_item[item_id] = relative.as_posix()
                    public_assets[relative.as_posix()] = {
                        "item_id": item_id,
                        "sha256": _sha256_file(out_dir / relative),
                    }
            public_pairs.append({
                "pair_id": pair_id,
                "task_prompt": pair["task_prompt"],
                "rubric": rubric,
                "artifact_a": asset_for_item[a_item],
                "artifact_b": asset_for_item[b_item],
                "allowed_choices": ["A", "B", "tie", "abstain"],
            })
        packet = {
            "schema_version": GAUNTLET_SCHEMA_VERSION,
            "study_id": study_id,
            "form_id": form_name,
            "instructions": (
                "Judge only the displayed behavior against the task prompt and "
                "rubric. Choose A or B, use tie only for indistinguishable task "
                "performance, and abstain when evidence is insufficient."
            ),
            "blinding": {
                "counterbalanced_forms": forms,
                "hidden_fields": [
                    "source identity", "condition", "iteration", "reward code",
                    "evaluator score", "competence label", "expected preference",
                ],
            },
            "pairs": public_pairs,
        }
        packet_name = f"study_packet_form_{form_name}.json"
        write_json_atomic(out_dir / packet_name, packet)
        with (out_dir / f"response_template_form_{form_name}.jsonl").open(
            "w", encoding="utf-8",
        ) as handle:
            for pair in public_pairs:
                handle.write(json.dumps({
                    "study_id": study_id,
                    "form_id": form_name,
                    "pair_id": pair["pair_id"],
                    "rater_id": "",
                    "choice": "",
                    "confidence": None,
                    "duration_seconds": None,
                    "notes": "",
                }, sort_keys=True) + "\n")
        public_packets[form_name] = {
            "path": packet_name,
            "sha256": _sha256_json(packet),
            "n_presentations": len(public_pairs),
        }

    private_items = {
        item_id: {
            **item,
            "artifact_path": str(item["artifact_path"]),
        }
        for item_id, item in item_by_id.items()
    }
    key = _key_with_hash({
        "schema_version": GAUNTLET_SCHEMA_VERSION,
        "study_id": study_id,
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": _sha256_file(source_manifest),
        "evaluator": manifest.get("evaluator"),
        "design": {
            "seed": int(seed),
            "forms": forms,
            "max_pairs_per_group": int(max_pairs_per_group),
            "reliability_repeats": int(reliability_repeats),
            "evaluator_tie_band": tie_band,
            "pairing": "balanced_round_robin_over_behavior_class_pairs",
            "position_assignment": "random_form_A_counterbalanced_form_B",
            "media_policy": "video_only_metadata_stripped_no_audio",
        },
        "public_packets": public_packets,
        "public_assets": public_assets,
        "items": private_items,
        "pairs": private_pairs,
    })
    write_json_atomic(out_dir / "study_key.json", key)
    summary = {
        "study_id": study_id,
        "out_dir": str(out_dir),
        "n_source_items": len(items),
        "n_primary_pairs": len(base_pairs),
        "n_reliability_repeats": reliability_repeats,
        "forms": form_names,
        "presentations_per_form": len(pair_order),
        "study_key_sha256": key["key_sha256"],
    }
    write_json_atomic(out_dir / "build_summary.json", summary)
    return summary


def _load_responses(
    path: Path, key: Mapping[str, Any],
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    rater_forms: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GauntletError(
                    f"responses line {line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise GauntletError(f"responses line {line_number} must be an object")
            if raw.get("study_id") != key["study_id"]:
                raise GauntletError(
                    f"responses line {line_number} has the wrong study_id"
                )
            form = _require_text(raw, "form_id", f"responses[{line_number}]")
            pair_id = _require_text(raw, "pair_id", f"responses[{line_number}]")
            rater = _require_text(raw, "rater_id", f"responses[{line_number}]")
            choice_raw = _require_text(raw, "choice", f"responses[{line_number}]")
            choice = choice_raw.upper() if choice_raw.upper() in {"A", "B"} else choice_raw.lower()
            if choice not in _CHOICES:
                raise GauntletError(
                    f"responses line {line_number} choice must be one of "
                    f"{sorted(_CHOICES)}"
                )
            if pair_id not in key["pairs"]:
                raise GauntletError(
                    f"responses line {line_number} has unknown pair_id {pair_id!r}"
                )
            presentation = key["pairs"][pair_id]["presentations"]
            if form not in presentation:
                raise GauntletError(
                    f"responses line {line_number} pair {pair_id!r} is not in form {form!r}"
                )
            identity = (rater, form, pair_id)
            if identity in seen:
                raise GauntletError(
                    f"duplicate response for rater/form/pair: {identity}"
                )
            seen.add(identity)
            if rater in rater_forms and rater_forms[rater] != form:
                raise GauntletError(
                    f"rater {rater!r} appears in multiple counterbalanced forms"
                )
            rater_forms[rater] = form
            confidence = raw.get("confidence")
            if confidence is not None:
                confidence = _finite_number(
                    confidence, f"responses[{line_number}].confidence",
                )
                if not 0 <= confidence <= 1:
                    raise GauntletError("confidence must be in [0, 1]")
            responses.append({
                **raw,
                "form_id": form,
                "pair_id": pair_id,
                "rater_id": rater,
                "choice": choice,
                "confidence": confidence,
            })
    if not responses:
        raise GauntletError("responses file contains no responses")
    return responses


def _chosen_item(response: Mapping[str, Any], key: Mapping[str, Any]) -> str:
    choice = str(response["choice"])
    if choice in {"tie", "abstain"}:
        return choice
    side = "a_item_id" if choice == "A" else "b_item_id"
    return str(key["pairs"][response["pair_id"]]["presentations"]
               [response["form_id"]][side])


def _majority(labels: Sequence[str]) -> tuple[str, float]:
    if not labels:
        return "inconclusive", 0.0
    counts = Counter(labels)
    best = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == best)
    if len(winners) != 1:
        return "inconclusive", best / len(labels)
    return winners[0], best / len(labels)


def _mean(values: Sequence[float]) -> float:
    return (
        sum(float(v) for v in values) / len(values)
        if len(values) else float("nan")
    )


def _accuracy(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"point": None, "ci_low": None, "ci_high": None, "n": 0}
    return stratified_bootstrap_ci(
        values, statistic=_mean, n_boot=2000, alpha=0.05, rng_seed=0,
    )


def _wilson(successes: int, total: int) -> dict[str, Any]:
    if total == 0:
        return {"rate": None, "ci_low": None, "ci_high": None, "n": 0}
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "rate": p, "ci_low": max(0.0, center - radius),
        "ci_high": min(1.0, center + radius), "n": total,
    }


def _krippendorff_alpha_nominal(units: Mapping[str, Sequence[str]]) -> Optional[float]:
    usable = [list(labels) for labels in units.values() if len(labels) >= 2]
    if not usable:
        return None
    observed_disagreements = sum(
        sum(
            labels[i] != labels[j]
            for i in range(len(labels))
            for j in range(len(labels))
            if i != j
        )
        for labels in usable
    )
    observed_pairs = sum(len(labels) * (len(labels) - 1) for labels in usable)
    do = observed_disagreements / observed_pairs if observed_pairs else 0.0
    pooled = Counter(label for labels in usable for label in labels)
    n = sum(pooled.values())
    if n < 2:
        return None
    de = sum(count * (n - count) for count in pooled.values()) / (n * (n - 1))
    if de == 0:
        return 1.0 if do == 0 else None
    return 1.0 - do / de


def _dimension_breakdown(
    primary_pairs: Sequence[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
    human_majority: Mapping[str, str],
    evaluator_choice: Mapping[str, Optional[str]],
    field: str,
) -> dict[str, Any]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in primary_pairs:
        values = sorted({str(items[item_id][field]) for item_id in pair["item_ids"]})
        buckets[" + ".join(values)].append(pair)
    out: dict[str, Any] = {}
    for value, pairs in sorted(buckets.items()):
        human = [
            float(human_majority.get(pair["pair_id"]) == pair["expected_item_id"])
            for pair in pairs
            if human_majority.get(pair["pair_id"]) not in {None, "tie", "inconclusive"}
        ]
        evaluator = [
            float(evaluator_choice.get(pair["pair_id"]) == pair["expected_item_id"])
            for pair in pairs
            if evaluator_choice.get(pair["pair_id"]) not in {None, "tie"}
        ]
        out[value] = {
            "n_pairs": len(pairs),
            "human_majority_accuracy": _accuracy(human),
            "evaluator_accuracy": _accuracy(evaluator),
        }
    return out


def analyze_blind_study(
    study_key: Path, responses_path: Path, out_dir: Path,
) -> dict[str, Any]:
    """Validate responses and write a pre-specified evaluator/human analysis."""
    key = load_and_verify_study_key(study_key)
    responses = _load_responses(responses_path, key)
    items = key["items"]
    pairs = key["pairs"]
    primary_pairs = [
        pair for pair in pairs.values()
        if pair["presentation_kind"] == "primary"
    ]
    tie_band = float(key["design"]["evaluator_tie_band"])
    evaluator_metadata = key.get("evaluator") or {}
    higher_is_better = bool(evaluator_metadata.get("higher_is_better", True))

    canonical_by_response: dict[tuple[str, str], str] = {}
    primary_labels: dict[str, list[str]] = defaultdict(list)
    agreement_labels: dict[str, list[str]] = defaultdict(list)
    response_accuracy: list[float] = []
    abstentions = ties = 0
    for response in responses:
        chosen = _chosen_item(response, key)
        canonical_by_response[(response["rater_id"], response["pair_id"])] = chosen
        pair = pairs[response["pair_id"]]
        if chosen == "abstain":
            abstentions += 1
            continue
        if chosen == "tie":
            ties += 1
        if pair["presentation_kind"] == "primary":
            primary_labels[pair["pair_id"]].append(chosen)
            agreement_labels[pair["pair_id"]].append(
                "tie" if chosen == "tie"
                else "expected" if chosen == pair["expected_item_id"]
                else "other"
            )
            if chosen != "tie":
                response_accuracy.append(float(chosen == pair["expected_item_id"]))

    if not primary_labels:
        raise GauntletError(
            "responses contain no primary-pair judgments; reliability repeats "
            "cannot be analyzed without their primary comparisons"
        )

    majority: dict[str, str] = {}
    consensus: dict[str, float] = {}
    for pair in primary_pairs:
        label, strength = _majority(primary_labels.get(pair["pair_id"], []))
        majority[pair["pair_id"]] = label
        consensus[pair["pair_id"]] = strength

    evaluator_choice: dict[str, Optional[str]] = {}
    evaluator_expected_accuracy: list[float] = []
    evaluator_human_accuracy: list[float] = []
    human_pair_accuracy: list[float] = []
    false_competence: dict[str, list[float]] = defaultdict(list)
    pair_records: list[dict[str, Any]] = []
    for pair in primary_pairs:
        first_id, second_id = pair["item_ids"]
        first_score = items[first_id].get("evaluator_score")
        second_score = items[second_id].get("evaluator_score")
        evaluator: Optional[str]
        if first_score is None or second_score is None:
            evaluator = None
        elif abs(float(first_score) - float(second_score)) <= tie_band:
            evaluator = "tie"
        else:
            first_wins = float(first_score) > float(second_score)
            if not higher_is_better:
                first_wins = not first_wins
            evaluator = first_id if first_wins else second_id
        evaluator_choice[pair["pair_id"]] = evaluator
        expected = pair["expected_item_id"]
        human = majority[pair["pair_id"]]
        if human not in {"tie", "inconclusive"}:
            human_pair_accuracy.append(float(human == expected))
        if evaluator not in {None, "tie"}:
            evaluator_expected_accuracy.append(float(evaluator == expected))
            lower_id = first_id if expected == second_id else second_id
            lower_class = str(items[lower_id]["behavior_class"])
            false_competence[lower_class].append(float(evaluator == lower_id))
            if human not in {"tie", "inconclusive"}:
                evaluator_human_accuracy.append(float(evaluator == human))
        pair_records.append({
            "pair_id": pair["pair_id"],
            "comparison_group": pair["comparison_group"],
            "expected_item_id": expected,
            "evaluator_preference": evaluator,
            "human_majority_preference": human,
            "human_consensus": consensus[pair["pair_id"]],
            "n_human_labels": len(primary_labels.get(pair["pair_id"], [])),
        })

    repeat_checks: list[float] = []
    primary_for_base = {
        pair["base_pair_id"]: pair["pair_id"] for pair in primary_pairs
    }
    for pair in pairs.values():
        if pair["presentation_kind"] != "reliability_repeat":
            continue
        primary_id = primary_for_base[pair["base_pair_id"]]
        for rater in {response["rater_id"] for response in responses}:
            a = canonical_by_response.get((rater, primary_id))
            b = canonical_by_response.get((rater, pair["pair_id"]))
            if a is not None and b is not None and "abstain" not in {a, b}:
                repeat_checks.append(float(a == b))

    order_bias: dict[str, Any] = {}
    for form in sorted(key["public_packets"]):
        decisive = [
            response for response in responses
            if response["form_id"] == form and response["choice"] in {"A", "B"}
        ]
        order_bias[form] = _wilson(
            sum(response["choice"] == "A" for response in decisive), len(decisive),
        )
    decisive_all = [r for r in responses if r["choice"] in {"A", "B"}]
    order_bias["overall"] = _wilson(
        sum(response["choice"] == "A" for response in decisive_all),
        len(decisive_all),
    )

    dimensions = {
        field: _dimension_breakdown(
            primary_pairs, items, majority, evaluator_choice, field,
        )
        for field in ("task_id", "robot_id", "embodiment_family", "motion_family")
    }
    analysis: dict[str, Any] = {
        "schema_version": GAUNTLET_SCHEMA_VERSION,
        "study_id": key["study_id"],
        "study_key_sha256": key["key_sha256"],
        "response_file_sha256": _sha256_file(Path(responses_path)),
        "counts": {
            "responses": len(responses),
            "raters": len({response["rater_id"] for response in responses}),
            "primary_pairs": len(primary_pairs),
            "pairs_with_human_labels": sum(bool(v) for v in primary_labels.values()),
            "ties": ties,
            "abstentions": abstentions,
        },
        "human": {
            "individual_accuracy_vs_expected": _accuracy(response_accuracy),
            "pair_majority_accuracy_vs_expected": _accuracy(human_pair_accuracy),
            "mean_pair_consensus": _mean([
                consensus[pair_id] for pair_id, labels in primary_labels.items()
                if labels
            ]),
            "krippendorff_alpha_nominal": _krippendorff_alpha_nominal(
                agreement_labels,
            ),
            "repeat_self_consistency": _accuracy(repeat_checks),
        },
        "evaluator": {
            "identity": evaluator_metadata,
            "pair_accuracy_vs_expected": _accuracy(evaluator_expected_accuracy),
            "pair_accuracy_vs_human_majority": _accuracy(evaluator_human_accuracy),
            "false_competence_by_lower_behavior_class": {
                behavior_class: _wilson(int(sum(values)), len(values))
                for behavior_class, values in sorted(false_competence.items())
            },
            "tie_band": tie_band,
        },
        "order_bias_display_a_choice_rate": order_bias,
        "dimensions": dimensions,
        "pairs": pair_records,
        "interpretation_limits": [
            "Bootstrap intervals use primary comparison pairs as units; raw "
            "responses are not treated as independent environment samples.",
            "Expected competence ranks are dataset labels, not a substitute for "
            "human authority; discrepancies remain visible at pair level.",
            "Evaluator-human accuracy excludes human-inconclusive and evaluator-tie pairs.",
        ],
    }
    analysis["analysis_sha256"] = _sha256_json(analysis)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "gauntlet_analysis.json", analysis)
    (out_dir / "gauntlet_analysis.md").write_text(
        _analysis_markdown(analysis), encoding="utf-8",
    )
    return analysis


def _format_rate(metric: Mapping[str, Any]) -> str:
    point = metric.get("point", metric.get("rate"))
    if point is None:
        return "not estimable (n=0)"
    return (
        f"{float(point):.3f} "
        f"[{float(metric['ci_low']):.3f}, {float(metric['ci_high']):.3f}] "
        f"(n={metric['n']})"
    )


def _analysis_markdown(analysis: Mapping[str, Any]) -> str:
    human = analysis["human"]
    evaluator = analysis["evaluator"]
    lines = [
        f"# Metric gauntlet analysis: {analysis['study_id']}",
        "",
        f"Analysis SHA-256: `{analysis['analysis_sha256']}`",
        "",
        "## Primary results",
        "",
        "| Measure | Estimate |",
        "|---|---:|",
        "| Human pair-majority accuracy vs expected | "
        f"{_format_rate(human['pair_majority_accuracy_vs_expected'])} |",
        "| Evaluator accuracy vs expected | "
        f"{_format_rate(evaluator['pair_accuracy_vs_expected'])} |",
        "| Evaluator agreement with human majority | "
        f"{_format_rate(evaluator['pair_accuracy_vs_human_majority'])} |",
        "| Human repeat self-consistency | "
        f"{_format_rate(human['repeat_self_consistency'])} |",
        "",
        f"Krippendorff's nominal alpha: `{human['krippendorff_alpha_nominal']}`  ",
        f"Mean pair consensus: `{human['mean_pair_consensus']:.3f}`",
        "",
        "## False competence by exploit/behavior class",
        "",
        "| Lower-ranked behavior class | Evaluator chose it |",
        "|---|---:|",
    ]
    for behavior_class, metric in evaluator[
        "false_competence_by_lower_behavior_class"
    ].items():
        lines.append(f"| {behavior_class} | {_format_rate(metric)} |")
    lines.extend([
        "",
        "## Audit notes",
        "",
        *[f"- {note}" for note in analysis["interpretation_limits"]],
        "",
    ])
    return "\n".join(lines)
