"""Policy export — self-contained deployment bundles for sim-to-real.

A bundle is a single zip that carries everything needed to reproduce or
deploy one trained iteration:

    policy_<project>_iter<N>.zip
    ├── manifest.json        # schema, provenance, dims, hashes
    ├── checkpoint.pt|zip    # the raw trained checkpoint, byte-exact
    ├── policy.onnx          # actor network, ONNX (best-effort)
    ├── policy_ts.pt         # actor network, TorchScript (best-effort)
    ├── reward/
    │   ├── reward_spec.json # REWARD_SPEC snapshot the iter trained under
    │   └── v<n>.py          # exact reward source (when resolvable)
    ├── env_spec.json        # env spec the iter trained under (see manifest
    │                        #   env_spec_source for how exact the match is)
    ├── config.toml          # project config snapshot
    ├── metrics.json         # the iter's training/eval metrics
    ├── run_context.json     # package versions / SHAs / seeds (if recorded)
    └── DEPLOY.md            # loading recipes (ONNX / TorchScript / raw)

Checkpoint formats understood:

* mjlab / rsl_rl ``checkpoint.pt`` — ``{actor_state_dict, critic_state_dict,
  optimizer_state_dict, iter, infos}`` where the actor is an MLP stored
  under ``mlp.<i>.{weight,bias}`` plus ``distribution.std_param``. The
  actor architecture is reconstructed *from the state dict itself* (layer
  shapes are ground truth); the activation comes from the mjlab task cfg
  when importable, else defaults to ``elu`` and the manifest says so.
* stable-baselines3 ``checkpoint.zip`` — bundled byte-exact; ONNX /
  TorchScript export is attempted via the documented SB3 recipe when
  stable_baselines3 is importable.

Everything network-related is best-effort by design: the raw checkpoint +
DEPLOY.md recipe are always present, so a bundle is never useless even
when torch/onnx/sb3 are missing from the environment doing the export.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

EXPORT_SCHEMA_VERSION = 1

_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")
_MLP_KEY_RE = re.compile(r"^mlp\.(\d+)\.(weight|bias)$")


class ExportError(RuntimeError):
    """Raised when a bundle cannot be built at all (no checkpoint etc.)."""


@dataclass
class ExportResult:
    bundle_path: Path
    manifest: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


# ── discovery ──────────────────────────────────────────────────────────────

def list_exportable_iters(runs_root: Path | str) -> list[dict[str, Any]]:
    """Iterations under ``runs_root`` that have a trained checkpoint on disk.

    Returns one dict per iteration, sorted by index:
    ``{iter_index, checkpoint, checkpoint_bytes, primary_metric, fitness,
    reward_version}`` — the scalar fields are None when the sidecar files
    are missing/unreadable. Disk is the source of truth (JobManager runs
    are in-memory and vanish on backend restart; these artifacts don't).
    """
    root = Path(runs_root)
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        m = _ITER_DIR_RE.match(d.name)
        if not m or not d.is_dir():
            continue
        ckpt = _find_checkpoint(d)
        if ckpt is None:
            continue
        metrics = _load_json(d / "metrics.json") or {}
        spec = _load_json(d / "reward_spec.json") or {}
        behavior = _load_json(d / "rollout" / "behavior.json") or {}
        fitness_doc = _load_json(d / "fitness.json") or {}
        primary = None
        mm = metrics.get("metrics")
        if isinstance(mm, dict):
            v = mm.get("mean_return")
            if isinstance(v, (int, float)):
                primary = float(v)
        # Current objective-fitness runs persist the firewall score beside
        # the iteration in fitness.json.  behavior.json is rollout telemetry
        # and only older runs stored a fitness scalar there.  Prefer the
        # authoritative current file while retaining the legacy fallback.
        fitness = fitness_doc.get("fitness")
        if not isinstance(fitness, (int, float)):
            fitness = behavior.get("fitness")
        out.append({
            "iter_index": int(m.group(1)),
            "checkpoint": ckpt.name,
            "checkpoint_bytes": ckpt.stat().st_size,
            "primary_metric": primary,
            "fitness": fitness if isinstance(fitness, (int, float)) else None,
            "reward_version": spec.get("version"),
        })
    out.sort(key=lambda r: r["iter_index"])
    return out


def _find_checkpoint(iter_dir: Path) -> Optional[Path]:
    for name in ("checkpoint.pt", "checkpoint.zip"):
        p = iter_dir / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


# ── bundle builder ─────────────────────────────────────────────────────────

def export_policy_bundle(
    project_dir: Path | str,
    *,
    iter_index: Optional[int] = None,
    runs_root: Path | str | None = None,
    out_path: Path | str | None = None,
) -> ExportResult:
    """Build a self-contained deployment bundle for one trained iteration.

    ``iter_index=None`` picks the latest iteration that has a checkpoint.
    ``runs_root`` defaults to ``<project>/runs`` (mission stages pass their
    own ``.missions/<m>/stages/<s>/runs``). ``out_path`` defaults to
    ``<project>/exports/policy_<project>_iter<N>.zip``.
    """
    project = Path(project_dir).resolve()
    root = Path(runs_root).resolve() if runs_root else project / "runs"

    avail = list_exportable_iters(root)
    if not avail:
        raise ExportError(f"no exportable iterations under {root}")
    if iter_index is None:
        iter_index = avail[-1]["iter_index"]
    if not any(r["iter_index"] == iter_index for r in avail):
        raise ExportError(
            f"iter {iter_index} has no checkpoint under {root} "
            f"(available: {[r['iter_index'] for r in avail]})")

    iter_dir = root / f"iter_{iter_index}"
    ckpt = _find_checkpoint(iter_dir)
    assert ckpt is not None  # guaranteed by list_exportable_iters
    warnings: list[str] = []

    if out_path is None:
        exports_dir = project / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        out = exports_dir / f"policy_{project.name}_iter{iter_index}.zip"
    else:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="rs_export_") as td:
        stage = Path(td)

        files: list[tuple[Path, str]] = [(ckpt, ckpt.name)]

        # Reward: spec snapshot + exact source when resolvable.
        reward_spec = _load_json(iter_dir / "reward_spec.json")
        reward_version = (reward_spec or {}).get("version")
        if reward_spec is not None:
            files.append((iter_dir / "reward_spec.json",
                          "reward/reward_spec.json"))
        else:
            warnings.append("reward_spec.json missing from the iter dir")
        if isinstance(reward_version, str):
            # reward_spec.json is on-disk data — validate before it becomes
            # a path component (a hostile "../../x" must not read outside
            # the project once this is HTTP-triggered).
            if re.fullmatch(r"v\d+", reward_version):
                cand = project / "rewards" / f"{reward_version}.py"
                if cand.is_file():
                    files.append((cand, f"reward/{cand.name}"))
                else:
                    warnings.append(
                        f"reward source {cand.name} not found under rewards/")
            else:
                warnings.append(
                    f"reward version {reward_version!r} is not v<n> — "
                    "reward source not bundled")

        # Env spec: exact per-iter snapshot when present (written at train
        # time), else the project's current spec, flagged as approximate.
        env_spec_source = None
        iter_env = iter_dir / "env_spec.json"
        cur_env = project / "env" / "current.json"
        if iter_env.is_file():
            files.append((iter_env, "env_spec.json"))
            env_spec_source = "iter_snapshot"
        elif cur_env.is_file():
            files.append((cur_env, "env_spec.json"))
            env_spec_source = "project_current"
            warnings.append(
                "env_spec.json is the project's CURRENT spec — this run "
                "predates per-iter env snapshots, so the exact trained "
                "version is not guaranteed")

        # Project config + iter metrics + reproducibility context.
        for src, arc in (
            (project / "config.toml", "config.toml"),
            (iter_dir / "metrics.json", "metrics.json"),
            (iter_dir / "run_context.json", "run_context.json"),
        ):
            if src.is_file():
                files.append((src, arc))
            elif arc == "config.toml":
                warnings.append("config.toml missing from the project dir")

        # Neural-net exports (best-effort, never fatal).
        net_meta: dict[str, Any] = {}
        if ckpt.suffix == ".pt":
            net_meta = _export_rsl_rl_actor(
                ckpt, project, stage, files, warnings)
        elif ckpt.suffix == ".zip":
            net_meta = _export_sb3_actor(ckpt, stage, files, warnings)

        manifest: dict[str, Any] = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "kind": "reward-sculptor-policy-bundle",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sculptor_version": _sculptor_version(),
            "project": project.name,
            "iter_index": iter_index,
            "checkpoint": {
                "file": ckpt.name,
                "format": "rsl_rl" if ckpt.suffix == ".pt" else "sb3",
                "sha256": _sha256(ckpt),
            },
            "reward_version": reward_version,
            "env_spec_source": env_spec_source,
            "network": net_meta,
            "warnings": warnings,
        }

        deploy_md = _render_deploy_md(manifest)
        (stage / "DEPLOY.md").write_text(deploy_md, encoding="utf-8")
        files.append((stage / "DEPLOY.md", "DEPLOY.md"))

        # File table with hashes (manifest lists everything but itself).
        manifest["files"] = [
            {"path": arc, "sha256": _sha256(src), "bytes": src.stat().st_size}
            for src, arc in files
        ]

        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8")

        tmp_zip = stage / "bundle.zip"
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, "manifest.json")
            for src, arc in files:
                zf.write(src, arc)
        # Atomic-ish move into place (same-filesystem rename when possible).
        shutil.move(str(tmp_zip), str(out))

    return ExportResult(bundle_path=out, manifest=manifest, warnings=warnings)


# ── rsl_rl (.pt) actor reconstruction ──────────────────────────────────────

def _export_rsl_rl_actor(
    ckpt: Path, project: Path, stage: Path,
    files: list[tuple[Path, str]], warnings: list[str],
) -> dict[str, Any]:
    """Rebuild the actor MLP from the checkpoint state dict and export it.

    Layer sizes come from the state dict (ground truth). The activation
    comes from the mjlab task cfg when resolvable; else assumed ``elu``
    (rsl_rl's default) and flagged in the manifest.
    """
    try:
        import torch
    except ImportError:
        warnings.append("torch not importable — raw checkpoint only")
        return {}

    try:
        try:
            payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 — older ckpts pickle extras in infos
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 — corrupt ckpt must not kill export
        warnings.append(f"checkpoint unreadable ({type(e).__name__}: {e}) "
                        "— raw checkpoint only")
        return {}
    if not isinstance(payload, dict):
        warnings.append(
            f"checkpoint is a {type(payload).__name__}, not the rsl_rl dict "
            "format — raw checkpoint only")
        return {}
    actor_sd = payload.get("actor_state_dict")
    if not isinstance(actor_sd, dict):
        warnings.append("no actor_state_dict in checkpoint — raw only "
                        f"(keys: {sorted(payload)})")
        return {}

    # Observation normalization is part of the policy function. When the
    # checkpoint carries rsl_rl EmpiricalNormalization state we BAKE it
    # into the exported graph — y = (x - mean) / (std + eps), eps=1e-2,
    # applied before the MLP exactly as rsl_rl's MLPModel does. Unknown
    # normalizer shapes refuse the nn export (a bare-MLP export taking
    # raw obs where training used normalized ones is silently wrong).
    norm = _extract_obs_normalizer(payload, actor_sd, warnings)
    if norm == "refuse":
        return {"obs_normalization": True}

    layers = _mlp_layers_from_state_dict(actor_sd)
    if not layers:
        warnings.append("actor_state_dict has no mlp.<i> Linear layers — "
                        "unsupported architecture, raw checkpoint only")
        return {}
    reason = _mlp_structure_problem(actor_sd, layers)
    if reason:
        warnings.append(
            f"actor mlp structure not reconstructable ({reason}) — "
            "raw checkpoint only")
        return {}

    obs_dim = layers[0][1]
    act_dim = layers[-1][2]
    hidden = [out for (_, _, out) in layers[:-1]]

    activation, activation_assumed = _resolve_activation(project, warnings)
    if activation not in _ACTIVATIONS:
        # Never guess: an unmapped activation (lrelu/silu/mish/crelu...)
        # built as ELU would ship a silently wrong policy that even the
        # trace parity check can't catch (it compares against the same
        # wrongly-built module).
        warnings.append(
            f"activation {activation!r} from the task cfg has no known "
            "torch equivalent here — raw checkpoint only")
        return {}

    model = _build_mlp(layers, activation, norm=norm)
    # Load only the mlp.* weights; distribution params are not part of the
    # deterministic deployment path (we export the mean action).
    mlp_sd = {k: v for k, v in actor_sd.items() if _MLP_KEY_RE.match(k)}
    try:
        model.mlp.load_state_dict(
            {k.removeprefix("mlp."): v for k, v in mlp_sd.items()},
            strict=True)
    except Exception as e:  # noqa: BLE001 — degrade, never die
        warnings.append(
            f"actor weights did not load into the reconstructed MLP "
            f"({type(e).__name__}: {e}) — raw checkpoint only")
        return {}
    model.eval()

    if norm is not None and not _verify_normalizer_parity(
            model, norm, obs_dim, warnings):
        return {"obs_normalization": True}

    meta: dict[str, Any] = {
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "hidden_dims": hidden,
        "activation": activation,
        "activation_assumed": activation_assumed,
        "output": "mean_action",
        "obs_normalization_baked": norm is not None,
        "exports": [],
    }

    example = torch.zeros(1, obs_dim)

    # TorchScript.
    try:
        with torch.no_grad():
            ts = torch.jit.trace(model, example)
            # Sanity: traced output must match the eager module.
            if not torch.allclose(ts(example), model(example), atol=1e-6):
                raise RuntimeError("traced output != eager output")
        ts_path = stage / "policy_ts.pt"
        ts.save(str(ts_path))
        files.append((ts_path, "policy_ts.pt"))
        meta["exports"].append("policy_ts.pt")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"TorchScript export failed ({type(e).__name__}: {e})")

    # ONNX.
    try:
        onnx_path = stage / "policy.onnx"
        _onnx_export(model, example, onnx_path)
        files.append((onnx_path, "policy.onnx"))
        meta["exports"].append("policy.onnx")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"ONNX export failed ({type(e).__name__}: {e})")

    return meta


def _mlp_layers_from_state_dict(sd: dict) -> list[tuple[int, int, int]]:
    """``[(seq_index, in_features, out_features), ...]`` sorted by index."""
    layers = []
    for k, v in sd.items():
        m = _MLP_KEY_RE.match(k)
        if m and m.group(2) == "weight" and getattr(v, "ndim", 0) == 2:
            layers.append((int(m.group(1)), int(v.shape[1]), int(v.shape[0])))
    layers.sort()
    return layers


def _mlp_structure_problem(
    actor_sd: dict, layers: list[tuple[int, int, int]],
) -> str | None:
    """Reject state dicts the Linear+activation rebuild can't represent.

    None = fine. A reason string = bail to raw-only. Guards:
    * a parameterized ``mlp.<i>`` module that is NOT a 2-D Linear weight
      (e.g. LayerNorm) would either strict-load-fail or be silently
      replaced by an activation;
    * consecutive Linear dims must chain (out_i == in_{i+1});
    * index gaps > 2 would make the rebuild stack multiple activations
      (ELU∘ELU ≠ ELU) where the original had dropout/identity modules.
    """
    linear_idx = {i for (i, _, _) in layers}
    for k in actor_sd:
        m = _MLP_KEY_RE.match(k)
        if m and int(m.group(1)) not in linear_idx:
            return f"parameterized non-Linear module at mlp.{m.group(1)}"
    for (i0, _, out0), (i1, in1, _) in zip(layers, layers[1:]):
        if out0 != in1:
            return f"dims don't chain (mlp.{i0} out={out0} vs mlp.{i1} in={in1})"
        if i1 - i0 > 2:
            return f"index gap {i1 - i0} between mlp.{i0} and mlp.{i1}"
    return None


_ACTIVATIONS = {
    "elu": "ELU", "relu": "ReLU", "tanh": "Tanh", "selu": "SELU",
    "leaky_relu": "LeakyReLU", "lrelu": "LeakyReLU", "sigmoid": "Sigmoid",
    "gelu": "GELU", "softplus": "Softplus", "silu": "SiLU", "mish": "Mish",
    # NOT here on purpose: "crelu" (doubles width — no module swap is right).
}


_RSL_RL_NORM_EPS = 1e-2  # EmpiricalNormalization default (not in the sd)


def _extract_obs_normalizer(payload: dict, actor_sd: dict, warnings: list):
    """Observation-normalizer stats to bake into the export.

    Returns None (no normalizer), a ``{"mean", "std"}`` dict of 1-D
    tensors, or the string ``"refuse"`` when normalizer-ish state exists
    but doesn't match the known rsl_rl EmpiricalNormalization shape —
    exporting a bare MLP in that case would be silently wrong.
    """
    known_prefix = "obs_normalizer."
    known = {"_mean", "_std", "_var", "count"}

    embedded = {
        k.removeprefix(known_prefix): v
        for k, v in actor_sd.items()
        if isinstance(k, str) and k.startswith(known_prefix)
    }
    payload_norm = payload.get("obs_norm_state_dict")
    other_norm_keys = [
        k for k in payload
        if isinstance(k, str) and "norm" in k.lower()
        and k != "obs_norm_state_dict"
    ] + [
        k for k in actor_sd
        if isinstance(k, str) and "norm" in k.lower()
        and not k.startswith(known_prefix)
    ]

    def _refuse(detail: str):
        warnings.append(
            f"checkpoint carries observation-normalizer state the exporter "
            f"doesn't recognise ({detail}) — a bare-MLP export would be "
            "numerically wrong, so only the raw checkpoint is bundled. "
            "Apply the normalizer stats before the MLP when deploying.")
        return "refuse"

    if other_norm_keys:
        return _refuse(f"keys: {sorted(set(other_norm_keys))}")

    source = None
    if embedded:
        if not set(embedded) <= known or not {"_mean", "_std"} <= set(embedded):
            return _refuse(f"obs_normalizer keys: {sorted(embedded)}")
        source = embedded
    elif payload_norm is not None:
        # Present but not a usable dict → refuse, never a bare-MLP export.
        if not (isinstance(payload_norm, dict) and payload_norm):
            return _refuse(
                f"obs_norm_state_dict is {type(payload_norm).__name__}"
                f"{' (empty)' if payload_norm == {} else ''}")
        stripped = {
            (k.removeprefix(known_prefix)
             if isinstance(k, str) else k): v
            for k, v in payload_norm.items()
        }
        if not {"_mean", "_std"} <= set(stripped):
            return _refuse(f"obs_norm_state_dict keys: {sorted(payload_norm)}")
        source = stripped
    if source is None:
        return None
    try:
        mean = source["_mean"].reshape(-1).to(dtype=_f32())
        std = source["_std"].reshape(-1).to(dtype=_f32())
    except Exception as e:  # noqa: BLE001 — non-tensor stats → raw-only
        return _refuse(f"normalizer stats not tensors ({type(e).__name__}: {e})")
    return {"mean": mean, "std": std}


def _f32():
    import torch

    return torch.float32


def _verify_normalizer_parity(
    model, norm: dict, obs_dim: int, warnings: list,
) -> bool:
    """When rsl_rl is importable, check the baked (x-mean)/(std+eps) against
    the REAL EmpiricalNormalization loaded with the same stats. Guards
    against the installed rsl_rl's DEFAULT eps/formula differing from the
    1e-2 baked here — a run trained with a non-default eps override is
    undetectable (eps is a constructor arg, not checkpointed)."""
    import torch

    try:
        from rsl_rl.modules.normalization import EmpiricalNormalization
    except ImportError:
        warnings.append(
            "rsl_rl not importable — baked obs normalization uses the "
            f"documented default eps={_RSL_RL_NORM_EPS} unverified")
        return True
    try:
        ref = EmpiricalNormalization(obs_dim)
        ref._mean = norm["mean"].reshape(1, -1).clone()
        ref._std = norm["std"].reshape(1, -1).clone()
        x = torch.randn(4, obs_dim, generator=torch.Generator().manual_seed(1))
        baked = (x - norm["mean"]) / (norm["std"] + _RSL_RL_NORM_EPS)
        if not torch.allclose(ref(x), baked, atol=1e-6):
            warnings.append(
                "baked obs normalization disagrees with rsl_rl's "
                "EmpiricalNormalization — raw checkpoint only")
            return False
    except Exception as e:  # noqa: BLE001 — parity check must not kill export
        warnings.append(
            f"obs-normalizer parity check errored ({type(e).__name__}: {e}) "
            "— raw checkpoint only")
        return False
    return True


def _build_mlp(
    layers: list[tuple[int, int, int]], activation: str,
    norm: "dict | None" = None,
):
    """nn.Sequential whose child indices mirror the checkpoint's ``mlp.<i>``
    keys, so ``load_state_dict`` maps 1:1 (activations occupy the gaps).
    ``norm`` bakes (x - mean) / (std + eps) in front of the MLP."""
    import torch.nn as nn

    act_cls = getattr(nn, _ACTIVATIONS.get(activation, "ELU"))
    max_idx = layers[-1][0]
    linear_at = {i: (fin, fout) for (i, fin, fout) in layers}
    mods = []
    for i in range(max_idx + 1):
        if i in linear_at:
            fin, fout = linear_at[i]
            mods.append(nn.Linear(fin, fout))
        else:
            mods.append(act_cls())
    model = nn.Sequential(*mods)
    # Wrap under attribute name "mlp" so keys line up.
    return _Actor(model, norm=norm)


def _resolve_activation(project: Path, warnings: list[str]) -> tuple[str, bool]:
    """(activation, assumed?) — exact from the mjlab task cfg when possible."""
    task_id = None
    try:
        try:
            import tomllib
        except ImportError:  # pragma: no cover - py<3.11
            import tomli as tomllib  # type: ignore[no-redef]
        with (project / "config.toml").open("rb") as f:
            cfg = tomllib.load(f)
        task_id = ((cfg.get("adapter") or {}).get("config") or {}).get("task_id")
    except Exception:  # noqa: BLE001
        pass
    if isinstance(task_id, str) and task_id:
        try:
            from mjlab.tasks.registry import load_rl_cfg

            rl_cfg = load_rl_cfg(task_id)
            actor_cfg = getattr(rl_cfg, "actor", None)
            act = getattr(actor_cfg, "activation", None)
            if isinstance(act, str) and act:
                return act, False
        except Exception as e:  # noqa: BLE001
            warnings.append(
                f"could not read activation from mjlab task cfg "
                f"({type(e).__name__}: {e}) — assuming elu")
    else:
        warnings.append("no task_id in config.toml — assuming elu activation")
    return "elu", True


def _onnx_export(model, example, onnx_path: Path) -> None:
    import torch

    try:
        torch.onnx.export(
            model, (example,), str(onnx_path),
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            dynamo=False,
        )
    except TypeError:
        # Older/newer torch without the dynamo kwarg.
        torch.onnx.export(
            model, (example,), str(onnx_path),
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        )
    # Structural validation when onnx is importable (onnxruntime not
    # required); a malformed file must not ship silently.
    try:
        import onnx

        onnx.checker.check_model(str(onnx_path))
    except ImportError:
        pass


# ── stable-baselines3 (.zip) ───────────────────────────────────────────────

def _export_sb3_actor(
    ckpt: Path, stage: Path,
    files: list[tuple[Path, str]], warnings: list[str],
) -> dict[str, Any]:
    """Best-effort ONNX/TorchScript for an SB3 PPO policy (documented SB3
    recipe: mlp_extractor → action_net, deterministic mean action)."""
    try:
        import torch
        from stable_baselines3 import PPO
    except ImportError:
        warnings.append("stable_baselines3/torch not importable — "
                        "raw checkpoint only")
        return {}
    try:
        model = PPO.load(str(ckpt), device="cpu")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"SB3 load failed ({type(e).__name__}: {e}) — "
                        "raw checkpoint only")
        return {}

    try:
        import gymnasium.spaces as _spaces
    except ImportError:  # pragma: no cover - gym classic
        import gym.spaces as _spaces  # type: ignore[no-redef]
    if not (isinstance(model.observation_space, _spaces.Box)
            and isinstance(model.action_space, _spaces.Box)):
        warnings.append(
            f"SB3 spaces {type(model.observation_space).__name__}/"
            f"{type(model.action_space).__name__} — only Box→Box policies "
            "get an ONNX/TorchScript export; raw checkpoint bundled")
        return {}

    policy = model.policy
    obs_dim = int(model.observation_space.shape[0])
    act_dim = int(model.action_space.shape[0])

    # predict(deterministic=True) clips the mean action to the space
    # bounds — the export must match it exactly.
    low_np, high_np = model.action_space.low, model.action_space.high
    import numpy as _np

    clip = bool(_np.isfinite(low_np).all() and _np.isfinite(high_np).all())
    low = torch.as_tensor(low_np, dtype=torch.float32)
    high = torch.as_tensor(high_np, dtype=torch.float32)

    class _SB3Actor(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p
            self.register_buffer("low", low)
            self.register_buffer("high", high)
            self.clip = clip

        def forward(self, obs):
            feats = self.p.extract_features(
                obs, self.p.features_extractor)
            latent_pi = self.p.mlp_extractor.forward_actor(feats)
            action = self.p.action_net(latent_pi)
            if self.clip:
                action = torch.clamp(action, self.low, self.high)
            return action

    actor = _SB3Actor(policy).eval()
    example = torch.zeros(1, obs_dim)
    meta: dict[str, Any] = {
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "output": "mean_action",
        "action_clipped_to_space": clip,
        "exports": [],
    }
    # VecNormalize stats live OUTSIDE checkpoint.zip. If the project has
    # them on disk we can't fold them in here — say so rather than let a
    # normalized-obs policy run on raw observations.
    if (ckpt.parent / "vecnormalize.pkl").is_file():
        warnings.append(
            "vecnormalize.pkl found next to the checkpoint — the exported "
            "network expects NORMALIZED observations; apply the "
            "VecNormalize stats at deployment")
    try:
        with torch.no_grad():
            ts = torch.jit.trace(actor, example)
        ts_path = stage / "policy_ts.pt"
        ts.save(str(ts_path))
        files.append((ts_path, "policy_ts.pt"))
        meta["exports"].append("policy_ts.pt")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"TorchScript export failed ({type(e).__name__}: {e})")
    try:
        onnx_path = stage / "policy.onnx"
        _onnx_export(actor, example, onnx_path)
        files.append((onnx_path, "policy.onnx"))
        meta["exports"].append("policy.onnx")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"ONNX export failed ({type(e).__name__}: {e})")
    return meta


# ── misc ───────────────────────────────────────────────────────────────────

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(p: Path) -> Optional[dict]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _sculptor_version() -> str:
    try:
        from importlib.metadata import version

        return version("reward-sculptor")
    except Exception:  # noqa: BLE001
        return "unknown"


def _render_deploy_md(manifest: dict[str, Any]) -> str:
    net = manifest.get("network") or {}
    fmt = manifest["checkpoint"]["format"]
    obs = net.get("obs_dim", "?")
    act = net.get("action_dim", "?")
    activation = net.get("activation", "elu")
    lines = [
        f"# Policy bundle — {manifest['project']} · iter {manifest['iter_index']}",
        "",
        f"Exported by Reward Sculptor {manifest['sculptor_version']} "
        f"on {manifest['created_at']}.",
        "",
        "| | |",
        "|---|---|",
        f"| Checkpoint format | `{fmt}` |",
        f"| Observation dim | `{obs}` |",
        f"| Action dim | `{act}` |",
        f"| Reward version | `{manifest.get('reward_version')}` |",
        f"| Env spec source | `{manifest.get('env_spec_source')}` |",
        "",
        "The network outputs the **mean action** of the Gaussian policy "
        "(deterministic deployment). Observation layout, scaling, and "
        "action post-processing must match training — see `env_spec.json`, "
        "`config.toml`, and the adapter's state schema.",
        "",
    ]
    if net.get("obs_normalization_baked") and net.get("exports"):
        lines += [
            "Observation normalization — `(x - mean) / (std + 0.01)`, the "
            "rsl_rl EmpiricalNormalization the policy trained with — is "
            "**baked into** `policy.onnx` / `policy_ts.pt`: feed them RAW "
            "observations. The raw checkpoint keeps the normalizer as "
            "separate state you must apply yourself.",
            "",
        ]
    if "policy.onnx" in (net.get("exports") or []):
        lines += [
            "## ONNX",
            "```python",
            "import onnxruntime as ort",
            "import numpy as np",
            'sess = ort.InferenceSession("policy.onnx")',
            f"obs = np.zeros((1, {obs}), dtype=np.float32)  # your observation",
            'action = sess.run(["action"], {"obs": obs})[0]',
            "```",
            "",
        ]
    if "policy_ts.pt" in (net.get("exports") or []):
        lines += [
            "## TorchScript",
            "```python",
            "import torch",
            'policy = torch.jit.load("policy_ts.pt").eval()',
            f"action = policy(torch.zeros(1, {obs}))",
            "```",
            "",
        ]
    if fmt == "rsl_rl":
        hidden = net.get("hidden_dims", [])
        lines += [
            "## Raw checkpoint (rsl_rl format)",
            "```python",
            "import torch",
            'ckpt = torch.load("checkpoint.pt", map_location="cpu",'
            " weights_only=False)",
            'actor_sd = ckpt["actor_state_dict"]  # mlp.<i>.weight/.bias',
            f"# MLP: {obs} -> {' -> '.join(str(h) for h in hidden)} -> {act},"
            f" activation={activation}",
            "# Or load into a full rsl_rl runner:",
            "#   runner.load(path, load_cfg={'actor': True, 'critic': True,",
            "#               'optimizer': False, 'iteration': False, 'rnd': False})",
            "```",
        ]
    else:
        lines += [
            "## Raw checkpoint (stable-baselines3)",
            "```python",
            "from stable_baselines3 import PPO",
            'model = PPO.load("checkpoint.zip", device="cpu")',
            "action, _ = model.predict(obs, deterministic=True)",
            "```",
        ]
    if manifest.get("warnings"):
        lines += ["", "## Export warnings", ""]
        lines += [f"- {w}" for w in manifest["warnings"]]
    lines.append("")
    return "\n".join(lines)


def _Actor(mlp, norm=None):  # noqa: N802 — lazy-torch factory
    """Deterministic actor head: obs -> mean action, weights under ``mlp``
    so checkpoint keys (``mlp.<i>.weight``) load 1:1. When ``norm`` is
    given, applies rsl_rl's (x - mean) / (std + eps) before the MLP."""
    import torch
    import torch.nn as nn

    class Actor(nn.Module):
        def __init__(self, seq: nn.Sequential):
            super().__init__()
            self.mlp = seq
            self.normalize = norm is not None
            if norm is not None:
                self.register_buffer("obs_mean", norm["mean"].clone())
                self.register_buffer("obs_std", norm["std"].clone())
            else:
                # Registered regardless so TorchScript tracing sees stable
                # attribute types; unused when normalize is False.
                self.register_buffer("obs_mean", torch.zeros(1))
                self.register_buffer("obs_std", torch.ones(1))

        def forward(self, obs):
            if self.normalize:
                obs = (obs - self.obs_mean) / (self.obs_std + _RSL_RL_NORM_EPS)
            return self.mlp(obs)

    return Actor(mlp)
