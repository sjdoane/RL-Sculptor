"""Pure, content-addressed receipts for files consumed by runner phases.

The runner may execute locally or through a mirrored remote workspace, so an
absolute pathname is not an identity.  These receipts bind the exact bytes and,
for a world selection, the complete content-addressed component set embedded in
the selection document.  Missing optional inputs are represented explicitly.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from sculptor.world.artifacts import canonical_json_bytes, sha256_bytes


ENVIRONMENT_ARTIFACT_SCHEMA = "reward-sculptor-environment-artifacts-v1"
REWARD_MODULE_ARTIFACT_SCHEMA = "reward-sculptor-reward-module-artifact-v1"
REWARD_SELECTOR_SCHEMA = 1
REWARD_SELECTOR_NAME = "SCULPTOR_REWARD_SELECTOR"
_REWARD_VERSION_RE = re.compile(r"^v\d+\.py$")
_LEGACY_REWARD_TARGET_RE = re.compile(r"/\s*(['\"])(v\d+\.py)\1")
_PHASE_KEYS = {
    "train": ("env_spec", "world_selection"),
    "rollout": ("env_spec", "eval_reset", "world_selection"),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _literal_reward_selector(source: str) -> dict[str, Any] | None:
    """Read the canonical selector declaration without executing code."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("reward module is not valid Python syntax") from exc
    values: list[Any] = []
    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == REWARD_SELECTOR_NAME
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == REWARD_SELECTOR_NAME
        ):
            value = node.value
        if value is not None:
            try:
                values.append(ast.literal_eval(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{REWARD_SELECTOR_NAME} must be literal data"
                ) from exc
    if not values:
        return None
    if len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError("reward selector declaration is ambiguous")
    return dict(values[0])


def capture_reward_module_artifact(
    path: str | Path,
) -> dict[str, Any]:
    """Bind a runner reward input to the immutable module it executes.

    Ordinary reward modules are direct inputs.  ``rewards/current.py`` is a
    mutable compatibility selector, so its own digest cannot identify the
    reward admitted in the immutable world tuple.  Canonical selectors carry
    a literal filename+digest receipt; historical core/UI selectors are
    admitted only through their single recognizable sibling ``vN.py`` target.
    Both loader and selected bytes are returned so callers can re-attest the
    complete mapping before and after execution while publishing the selected
    digest as ``reward_module_sha256``.
    """
    loader_path = Path(path).expanduser().resolve()
    if (
        not loader_path.is_file()
        or loader_path.is_symlink()
    ):
        raise ValueError(
            f"reward module must be a regular file: {loader_path}"
        )
    try:
        loader_bytes = loader_path.read_bytes()
        source = loader_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"cannot read reward module at {loader_path}: {exc}"
        ) from exc

    selector = _literal_reward_selector(source)
    selection_kind = "direct"
    declared_sha256: str | None = None
    selected_path = loader_path
    if selector is not None:
        if set(selector) != {"schema", "filename", "sha256"}:
            raise ValueError("reward selector fields are non-canonical")
        if selector.get("schema") != REWARD_SELECTOR_SCHEMA:
            raise ValueError("reward selector schema is unsupported")
        filename = selector.get("filename")
        declared_sha256 = selector.get("sha256")
        if (
            not isinstance(filename, str)
            or _REWARD_VERSION_RE.fullmatch(filename) is None
            or not _is_sha256(declared_sha256)
        ):
            raise ValueError("reward selector filename/digest is invalid")
        selected_path = (loader_path.parent / filename).resolve()
        selection_kind = "selector"
    else:
        legacy_targets = {
            match.group(2) for match in _LEGACY_REWARD_TARGET_RE.finditer(source)
        }
        if legacy_targets:
            if len(legacy_targets) != 1:
                raise ValueError("legacy reward selector target is ambiguous")
            selected_path = (
                loader_path.parent / next(iter(legacy_targets))
            ).resolve()
            selection_kind = "legacy_selector"
        elif "spec_from_file_location" in source:
            raise ValueError(
                "reward selector imports another module without an exact "
                "immutable target receipt"
            )

    if (
        selected_path.parent != loader_path.parent
        or not selected_path.is_file()
        or selected_path.is_symlink()
    ):
        raise ValueError(
            f"selected reward must be a regular sibling file: {selected_path}"
        )
    try:
        selected_bytes = selected_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read selected reward at {selected_path}: {exc}"
        ) from exc
    selected_sha256 = _sha256_bytes(selected_bytes)
    if declared_sha256 is not None and selected_sha256 != declared_sha256:
        raise ValueError(
            "selected reward bytes differ from the selector's immutable digest"
        )
    return {
        "schema": REWARD_MODULE_ARTIFACT_SCHEMA,
        "selection_kind": selection_kind,
        "loader": {
            "path": str(loader_path),
            "sha256": _sha256_bytes(loader_bytes),
        },
        "selected": {
            "path": str(selected_path),
            "sha256": selected_sha256,
        },
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _file_receipt(path: str | Path | None, *, label: str) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {"present": False, "sha256": None}
    resolved = Path(path).expanduser().resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label} at {resolved}: {exc}") from exc
    return {
        "present": True,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _world_selection_receipt(
    path: str | Path | None,
) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {"present": False, "sha256": None}
    resolved = Path(path).expanduser().resolve()
    try:
        selection_bytes = resolved.read_bytes()
        payload = json.loads(selection_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"world selection at {resolved} is not valid JSON: {exc}"
        ) from exc
    receipt = {
        "present": True,
        "sha256": hashlib.sha256(selection_bytes).hexdigest(),
    }
    if not isinstance(payload, dict):
        raise ValueError(f"world selection at {resolved} must be a JSON object")
    tuple_hash = payload.get("tuple_hash")
    refs = payload.get("refs")
    if not _is_sha256(tuple_hash) or not isinstance(refs, dict) or not refs:
        raise ValueError(
            f"world selection at {resolved} lacks a valid tuple_hash/refs set"
        )
    canonical_refs: dict[str, dict[str, Any]] = {}
    project_root = (
        resolved.parent.parent
        if resolved.parent.name == "env"
        else resolved.parent
    ).resolve()
    for key, raw in sorted(refs.items()):
        if not isinstance(key, str) or not key or not isinstance(raw, dict):
            raise ValueError(f"world selection at {resolved} has invalid refs")
        sha256 = raw.get("sha256")
        kind = raw.get("kind")
        version = raw.get("version")
        ref_path = raw.get("path")
        if (
            not _is_sha256(sha256)
            or not isinstance(kind, str)
            or not kind
            or not isinstance(version, str)
            or not version
            or not isinstance(ref_path, str)
            or not ref_path
        ):
            raise ValueError(
                f"world selection at {resolved} has an invalid {key!r} ref"
            )
        declared_path = Path(ref_path)
        if declared_path.is_absolute():
            raise ValueError(
                f"world selection {key!r} ref path must be project-relative"
            )
        try:
            component_path = (project_root / declared_path).resolve(strict=True)
            component_path.relative_to(project_root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"world selection {key!r} ref escapes or is unreadable: "
                f"{ref_path!r}"
            ) from exc
        if not component_path.is_file():
            raise ValueError(
                f"world selection {key!r} ref is not a regular file: "
                f"{ref_path!r}"
            )
        try:
            component_bytes = component_path.read_bytes()
            component_sha256 = hashlib.sha256(component_bytes).hexdigest()
        except OSError as exc:
            raise ValueError(
                f"cannot read world selection {key!r} ref {component_path}: {exc}"
            ) from exc
        if component_sha256 != sha256:
            raise ValueError(
                f"world selection {key!r} ref sha256 mismatch: declared "
                f"{sha256}, actual {component_sha256}"
            )
        canonical_refs[key] = {
            "kind": kind,
            "version": version,
            "path": ref_path,
            "sha256": sha256,
        }
    expected_tuple_hash = sha256_bytes(canonical_json_bytes(canonical_refs))
    if tuple_hash != expected_tuple_hash:
        raise ValueError(
            f"world selection at {resolved} tuple_hash mismatch: declared "
            f"{tuple_hash}, computed {expected_tuple_hash}"
        )
    return {
        **receipt,
        "tuple_hash": tuple_hash,
        "refs": canonical_refs,
    }


def capture_environment_artifacts(
    *,
    env_spec_path: str | Path | None = None,
    eval_reset_path: str | Path | None = None,
    world_selection_path: str | Path | None = None,
) -> dict[str, Any]:
    """Hash the exact optional environment inputs at one process boundary."""
    receipt = {
        "schema": ENVIRONMENT_ARTIFACT_SCHEMA,
        "env_spec": _file_receipt(env_spec_path, label="environment spec"),
        "eval_reset": _file_receipt(eval_reset_path, label="evaluation reset"),
        "world_selection": _world_selection_receipt(world_selection_path),
    }
    issues = validate_environment_artifacts(receipt)
    if issues:  # pragma: no cover - constructors above establish this invariant
        raise ValueError("invalid environment artifact receipt: " + "; ".join(issues))
    return receipt


def environment_artifacts_for_phase(
    receipt: dict[str, Any],
    phase: Literal["train", "rollout"],
) -> dict[str, Any]:
    """Project the full receipt onto files the named runner phase can read."""
    issues = validate_environment_artifacts(receipt)
    if issues:
        raise ValueError("invalid environment artifact receipt: " + "; ".join(issues))
    return {
        "schema": ENVIRONMENT_ARTIFACT_SCHEMA,
        **{key: receipt[key] for key in _PHASE_KEYS[phase]},
    }


def validate_environment_artifacts(
    receipt: Any,
    *,
    phase: Literal["train", "rollout"] | None = None,
) -> list[str]:
    if not isinstance(receipt, dict):
        return ["environment artifact receipt is missing"]
    issues: list[str] = []
    expected_keys = set(_PHASE_KEYS[phase] if phase is not None else (
        "env_spec", "eval_reset", "world_selection",
    ))
    if receipt.get("schema") != ENVIRONMENT_ARTIFACT_SCHEMA:
        issues.append("environment artifact schema is unsupported")
    if set(receipt) != {"schema", *expected_keys}:
        issues.append("environment artifact receipt has non-canonical keys")
    for key in sorted(expected_keys):
        artifact = receipt.get(key)
        if not isinstance(artifact, dict):
            issues.append(f"{key} receipt is missing")
            continue
        present = artifact.get("present")
        digest = artifact.get("sha256")
        if not isinstance(present, bool):
            issues.append(f"{key}.present is invalid")
        elif present and not _is_sha256(digest):
            issues.append(f"{key}.sha256 is missing/invalid")
        elif not present and digest is not None:
            issues.append(f"{key}.sha256 must be null when absent")
        if key != "world_selection":
            if set(artifact) != {"present", "sha256"}:
                issues.append(f"{key} receipt has non-canonical keys")
            continue
        if present:
            if set(artifact) != {"present", "sha256", "tuple_hash", "refs"}:
                issues.append("world_selection receipt has non-canonical keys")
            if not _is_sha256(artifact.get("tuple_hash")):
                issues.append("world_selection.tuple_hash is invalid")
            refs = artifact.get("refs")
            if not isinstance(refs, dict) or not refs:
                issues.append("world_selection.refs is missing")
            else:
                refs_are_canonical = True
                for ref_key, ref in refs.items():
                    if (
                        not isinstance(ref_key, str)
                        or not ref_key
                        or not isinstance(ref, dict)
                        or set(ref) != {"kind", "version", "path", "sha256"}
                        or not _is_sha256(ref.get("sha256"))
                        or not all(
                            isinstance(ref.get(field), str) and ref.get(field)
                            for field in ("kind", "version", "path")
                        )
                    ):
                        issues.append("world_selection.refs is invalid")
                        refs_are_canonical = False
                        break
                if refs_are_canonical:
                    canonical_refs = {
                        key: dict(value)
                        for key, value in sorted(refs.items())
                    }
                    expected_tuple_hash = sha256_bytes(
                        canonical_json_bytes(canonical_refs)
                    )
                    if artifact.get("tuple_hash") != expected_tuple_hash:
                        issues.append(
                            "world_selection.tuple_hash does not match refs"
                        )
        elif set(artifact) != {"present", "sha256"}:
            issues.append("absent world_selection receipt has non-canonical keys")
    return issues
