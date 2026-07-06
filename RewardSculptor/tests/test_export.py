"""tests/test_export.py — policy export bundles.

Exercises the bundle builder against hand-built projects on disk using the
real Go1 rsl_rl checkpoint fixture (network reconstruction + ONNX/
TorchScript) and degraded projects (missing sidecars, corrupt checkpoints)
to prove the raw-checkpoint fallback never dies. Numeric parity between
the TorchScript export and a hand-built reference MLP is asserted exactly
— a silently-wrong exported policy is the one failure mode this feature
must never have.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from sculptor.export import (
    ExportError,
    export_policy_bundle,
    list_exportable_iters,
)

FIXTURES = Path(__file__).parent / "fixtures"
GO1_CKPT = FIXTURES / "go1_smoke_checkpoint.pt"


# ── helpers ────────────────────────────────────────────────────────────────

def _make_project(
    tmp_path: Path,
    *,
    iters: list[int] = (0,),
    checkpoint: Path | None = GO1_CKPT,
    reward_version: str = "v1",
    with_env_current: bool = True,
    with_iter_env: bool = False,
    task_id: str | None = "Mjlab-Velocity-Flat-Unitree-Go1",
) -> Path:
    project = tmp_path / "proj"
    rewards = project / "rewards"
    rewards.mkdir(parents=True)
    (rewards / f"{reward_version}.py").write_text(
        f"REWARD_SPEC = {{'version': {reward_version!r}}}\n"
        "def compute_reward(state, action, next_state, info):\n"
        "    return 0.0, {}\n",
        encoding="utf-8")
    cfg = '[target]\nname = "proj"\n\n[adapter]\n' \
          'class = "sculptor.adapters.mjlab.MjlabAdapter"\n'
    if task_id:
        cfg += f'config = {{ task_id = "{task_id}" }}\n'
    (project / "config.toml").write_text(cfg, encoding="utf-8")
    env_spec = {
        "env_spec_version": 1,
        "meta": {"version": "v2"},
        "shared": {},
        "train": {},
    }
    if with_env_current:
        env = project / "env"
        env.mkdir()
        (env / "current.json").write_text(json.dumps(env_spec))
    for i in iters:
        it = project / "runs" / f"iter_{i}"
        it.mkdir(parents=True)
        if checkpoint is not None:
            shutil.copy(checkpoint, it / "checkpoint.pt")
        (it / "reward_spec.json").write_text(
            json.dumps({"version": reward_version, "author": "sculptor"}))
        (it / "metrics.json").write_text(
            json.dumps({"metrics": {"mean_return": 10.0 + i}}))
        if with_iter_env:
            (it / "env_spec.json").write_text(json.dumps(env_spec))
    return project


# ── discovery ──────────────────────────────────────────────────────────────

def test_list_exportable_iters_empty_and_missing(tmp_path):
    assert list_exportable_iters(tmp_path / "nope") == []
    (tmp_path / "runs").mkdir()
    assert list_exportable_iters(tmp_path / "runs") == []


def test_list_exportable_iters_skips_checkpointless(tmp_path):
    project = _make_project(tmp_path, iters=[0, 2])
    # iter_1 exists but has no checkpoint → excluded
    (project / "runs" / "iter_1").mkdir()
    rows = list_exportable_iters(project / "runs")
    assert [r["iter_index"] for r in rows] == [0, 2]
    assert rows[0]["checkpoint"] == "checkpoint.pt"
    assert rows[0]["reward_version"] == "v1"
    assert rows[1]["primary_metric"] == pytest.approx(12.0)


def test_list_exportable_iters_ignores_empty_checkpoint(tmp_path):
    project = _make_project(tmp_path, iters=[0], checkpoint=None)
    (project / "runs" / "iter_0" / "checkpoint.pt").touch()
    assert list_exportable_iters(project / "runs") == []


# ── bundle building (rsl_rl path, real fixture) ────────────────────────────

def test_export_bundle_contents_and_manifest(tmp_path):
    project = _make_project(tmp_path)
    res = export_policy_bundle(project)

    assert res.bundle_path.is_file()
    assert res.bundle_path.parent == project / "exports"
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert {
        "manifest.json", "checkpoint.pt", "policy.onnx", "policy_ts.pt",
        "reward/reward_spec.json", "reward/v1.py", "env_spec.json",
        "config.toml", "metrics.json", "DEPLOY.md",
    } <= names

    assert manifest["schema_version"] == 1
    assert manifest["iter_index"] == 0
    assert manifest["reward_version"] == "v1"
    assert manifest["checkpoint"]["format"] == "rsl_rl"
    net = manifest["network"]
    assert net["obs_dim"] == 48
    assert net["action_dim"] == 12
    assert net["hidden_dims"] == [512, 256, 128]
    assert net["output"] == "mean_action"
    # every listed file carries a sha256
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])


def test_export_picks_latest_iter_by_default(tmp_path):
    project = _make_project(tmp_path, iters=[0, 3, 7])
    res = export_policy_bundle(project)
    assert res.manifest["iter_index"] == 7
    assert res.bundle_path.name.endswith("_iter7.zip")


def test_export_explicit_iter_and_out_path(tmp_path):
    project = _make_project(tmp_path, iters=[0, 3])
    out = tmp_path / "elsewhere" / "bundle.zip"
    res = export_policy_bundle(project, iter_index=3, out_path=out)
    assert res.bundle_path == out
    assert out.is_file()
    assert res.manifest["iter_index"] == 3


def test_export_missing_iter_raises(tmp_path):
    project = _make_project(tmp_path, iters=[0])
    with pytest.raises(ExportError, match="iter 5 has no checkpoint"):
        export_policy_bundle(project, iter_index=5)


def test_export_no_iters_raises(tmp_path):
    project = _make_project(tmp_path, iters=[])
    with pytest.raises(ExportError, match="no exportable iterations"):
        export_policy_bundle(project)


def test_export_prefers_iter_env_snapshot(tmp_path):
    project = _make_project(tmp_path, with_iter_env=True)
    res = export_policy_bundle(project)
    assert res.manifest["env_spec_source"] == "iter_snapshot"
    assert not any("CURRENT spec" in w for w in res.warnings)


def test_export_falls_back_to_current_env_with_warning(tmp_path):
    project = _make_project(tmp_path, with_iter_env=False)
    res = export_policy_bundle(project)
    assert res.manifest["env_spec_source"] == "project_current"
    assert any("CURRENT spec" in w for w in res.warnings)


def test_export_without_env_spec_at_all(tmp_path):
    project = _make_project(tmp_path, with_env_current=False)
    res = export_policy_bundle(project)
    assert res.manifest["env_spec_source"] is None
    with zipfile.ZipFile(res.bundle_path) as zf:
        assert "env_spec.json" not in zf.namelist()


def test_export_runs_root_override(tmp_path):
    """Mission stages keep their runs under .missions/<m>/stages/<s>/runs."""
    project = _make_project(tmp_path, iters=[])
    stage_runs = project / ".missions" / "m1" / "stages" / "s1" / "runs"
    it = stage_runs / "iter_2"
    it.mkdir(parents=True)
    shutil.copy(GO1_CKPT, it / "checkpoint.pt")
    (it / "reward_spec.json").write_text(json.dumps({"version": "v1"}))
    res = export_policy_bundle(project, runs_root=stage_runs)
    assert res.manifest["iter_index"] == 2


# ── network reconstruction correctness ─────────────────────────────────────

def test_torchscript_matches_hand_built_reference(tmp_path):
    """The exported policy must be numerically identical to the checkpoint
    weights run through the known Go1 architecture (48→512→256→128→12, ELU)."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    project = _make_project(tmp_path)
    res = export_policy_bundle(project)
    with zipfile.ZipFile(res.bundle_path) as zf:
        zf.extractall(tmp_path / "unpacked")

    ts = torch.jit.load(str(tmp_path / "unpacked" / "policy_ts.pt")).eval()
    ckpt = torch.load(GO1_CKPT, map_location="cpu", weights_only=False)
    ref = nn.Sequential(
        nn.Linear(48, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12))
    ref.load_state_dict({
        k.removeprefix("mlp."): v
        for k, v in ckpt["actor_state_dict"].items()
        if k.startswith("mlp.")})
    obs = torch.randn(8, 48, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(ts(obs), ref(obs))


def test_onnx_structurally_valid(tmp_path):
    onnx = pytest.importorskip("onnx")
    project = _make_project(tmp_path)
    res = export_policy_bundle(project)
    with zipfile.ZipFile(res.bundle_path) as zf:
        zf.extractall(tmp_path / "unpacked")
    model = onnx.load(str(tmp_path / "unpacked" / "policy.onnx"))
    onnx.checker.check_model(model)
    assert model.graph.input[0].name == "obs"
    assert model.graph.output[0].name == "action"


# ── degraded checkpoints: raw-only fallback, never fatal ───────────────────

def test_corrupt_checkpoint_still_bundles_raw(tmp_path):
    project = _make_project(tmp_path, checkpoint=None)
    it = project / "runs" / "iter_0"
    (it / "checkpoint.pt").write_bytes(b"not a torch file at all")
    res = export_policy_bundle(project)
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
    assert "checkpoint.pt" in names
    assert "policy.onnx" not in names
    assert any("unreadable" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_non_mlp_actor_still_bundles_raw(tmp_path):
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    it = project / "runs" / "iter_0"
    torch.save(
        {"actor_state_dict": {"rnn.weight_ih_l0": torch.zeros(4, 4)}},
        it / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert any("unsupported architecture" in w for w in res.warnings)
    with zipfile.ZipFile(res.bundle_path) as zf:
        assert "checkpoint.pt" in zf.namelist()


def test_missing_task_id_assumes_elu(tmp_path):
    project = _make_project(tmp_path, task_id=None)
    res = export_policy_bundle(project)
    net = res.manifest["network"]
    assert net["activation"] == "elu"
    assert net["activation_assumed"] is True
    assert any("assuming elu" in w for w in res.warnings)


def test_missing_reward_source_warns(tmp_path):
    project = _make_project(tmp_path)
    (project / "rewards" / "v1.py").unlink()
    res = export_policy_bundle(project)
    assert any("not found under rewards/" in w for w in res.warnings)
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
    assert "reward/reward_spec.json" in names
    assert "reward/v1.py" not in names


def test_deploy_md_mentions_dims_and_recipes(tmp_path):
    project = _make_project(tmp_path)
    res = export_policy_bundle(project)
    with zipfile.ZipFile(res.bundle_path) as zf:
        md = zf.read("DEPLOY.md").decode("utf-8")
    assert "onnxruntime" in md
    assert "torch.jit.load" in md
    assert "mean action" in md
    assert "48" in md and "12" in md


# ── silently-wrong-policy guards (review findings 1-4) ─────────────────────

def test_non_dict_payload_bundles_raw(tmp_path):
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    torch.save(torch.zeros(3), project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert any("not the rsl_rl dict format" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_known_obs_normalizer_is_baked_into_export(tmp_path):
    """rsl_rl EmpiricalNormalization state embedded in the actor sd must be
    BAKED into the exported graph: TS(raw obs) == MLP(normalizer(raw obs))
    computed with the real rsl_rl module."""
    torch = pytest.importorskip("torch")
    rsl_norm = pytest.importorskip("rsl_rl.modules.normalization")
    import torch.nn as nn

    project = _make_project(tmp_path, checkpoint=None)
    ckpt = torch.load(GO1_CKPT, map_location="cpu", weights_only=False)
    gen = torch.Generator().manual_seed(7)
    mean = torch.randn(48, generator=gen)
    std = torch.rand(48, generator=gen) + 0.5
    ckpt["actor_state_dict"] = dict(ckpt["actor_state_dict"])
    ckpt["actor_state_dict"]["obs_normalizer._mean"] = mean.reshape(1, -1)
    ckpt["actor_state_dict"]["obs_normalizer._std"] = std.reshape(1, -1)
    ckpt["actor_state_dict"]["obs_normalizer._var"] = (std ** 2).reshape(1, -1)
    ckpt["actor_state_dict"]["obs_normalizer.count"] = torch.tensor(100)
    torch.save(ckpt, project / "runs" / "iter_0" / "checkpoint.pt")

    res = export_policy_bundle(project)
    assert res.manifest["network"]["obs_normalization_baked"] is True
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
        zf.extractall(tmp_path / "unpacked")
    assert {"policy.onnx", "policy_ts.pt"} <= names

    # Reference: REAL rsl_rl normalizer + hand-built MLP.
    ref_norm = rsl_norm.EmpiricalNormalization(48)
    ref_norm._mean = mean.reshape(1, -1).clone()
    ref_norm._std = std.reshape(1, -1).clone()
    ref_mlp = nn.Sequential(
        nn.Linear(48, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12))
    ref_mlp.load_state_dict({
        k.removeprefix("mlp."): v
        for k, v in ckpt["actor_state_dict"].items()
        if k.startswith("mlp.")})
    ts = torch.jit.load(str(tmp_path / "unpacked" / "policy_ts.pt")).eval()
    obs = torch.randn(8, 48, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.allclose(ts(obs), ref_mlp(ref_norm(obs)), atol=1e-6)


def test_unknown_normalizer_refuses_network_export(tmp_path):
    """Normalizer-ish state the exporter doesn't recognise → raw-only, never
    a bare-MLP guess."""
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    ckpt = torch.load(GO1_CKPT, map_location="cpu", weights_only=False)
    ckpt["critic_obs_norm_stats"] = {"mean": torch.zeros(48)}
    torch.save(ckpt, project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert res.manifest["network"] == {"obs_normalization": True}
    assert any("normalizer" in w for w in res.warnings)
    with zipfile.ZipFile(res.bundle_path) as zf:
        assert "policy.onnx" not in zf.namelist()
        assert "policy_ts.pt" not in zf.namelist()


def test_unknown_activation_refuses_network_export(tmp_path, monkeypatch):
    """An activation with no known torch equivalent must bail to raw-only —
    building with a guessed module ships a silently wrong policy."""
    import sculptor.export as ex

    monkeypatch.setattr(
        ex, "_resolve_activation", lambda project, warnings: ("crelu", False))
    project = _make_project(tmp_path)
    res = export_policy_bundle(project)
    assert any("no known torch equivalent" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_parameterized_non_linear_module_bails(tmp_path):
    """LayerNorm-style 1-D params at an mlp index must not be silently
    replaced by an activation."""
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    sd = {
        "mlp.0.weight": torch.zeros(8, 4), "mlp.0.bias": torch.zeros(8),
        "mlp.1.weight": torch.zeros(8),   # LayerNorm-ish
        "mlp.1.bias": torch.zeros(8),
        "mlp.2.weight": torch.zeros(2, 8), "mlp.2.bias": torch.zeros(2),
    }
    torch.save({"actor_state_dict": sd},
               project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert any("non-Linear module at mlp.1" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_unchained_dims_bail(tmp_path):
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    sd = {
        "mlp.0.weight": torch.zeros(8, 4), "mlp.0.bias": torch.zeros(8),
        "mlp.2.weight": torch.zeros(2, 16), "mlp.2.bias": torch.zeros(2),
    }
    torch.save({"actor_state_dict": sd},
               project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert any("dims don't chain" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_large_index_gap_bails(tmp_path):
    """A gap > 2 would stack multiple activations (ELU∘ELU ≠ ELU) where the
    original had dropout/identity modules — refuse instead."""
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    sd = {
        "mlp.0.weight": torch.zeros(8, 4), "mlp.0.bias": torch.zeros(8),
        "mlp.3.weight": torch.zeros(2, 8), "mlp.3.bias": torch.zeros(2),
    }
    torch.save({"actor_state_dict": sd},
               project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert any("index gap" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_traversal_reward_version_not_bundled(tmp_path):
    """A hostile reward_spec.json version must not become a path component."""
    project = _make_project(tmp_path)
    it = project / "runs" / "iter_0"
    secret = tmp_path / "secret.py"
    secret.write_text("SECRET = 1\n")
    (it / "reward_spec.json").write_text(
        json.dumps({"version": "../../secret"}))
    res = export_policy_bundle(project)
    assert any("not v<n>" in w for w in res.warnings)
    with zipfile.ZipFile(res.bundle_path) as zf:
        for name in zf.namelist():
            assert "secret" not in name
            if name.startswith("reward/") and name.endswith(".py"):
                raise AssertionError("no reward source should be bundled")


# ── sb3 path (offline; uses the hopper example checkpoint if present) ──────

HOPPER_CKPT = (
    Path(__file__).parent.parent
    / "examples" / "hopper" / "runs" / "iter_1" / "checkpoint.zip")


@pytest.mark.skipif(not HOPPER_CKPT.is_file(), reason="hopper example absent")
def test_sb3_bundle_exports_and_matches_predict(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")

    project = _make_project(tmp_path, checkpoint=None, task_id=None)
    it = project / "runs" / "iter_0"
    shutil.copy(HOPPER_CKPT, it / "checkpoint.zip")
    res = export_policy_bundle(project)
    assert res.manifest["checkpoint"]["format"] == "sb3"
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
        zf.extractall(tmp_path / "unpacked")
    assert "checkpoint.zip" in names
    if "policy_ts.pt" in names:  # best-effort export succeeded
        model = sb3.PPO.load(str(HOPPER_CKPT), device="cpu")
        obs = np.random.RandomState(0).randn(
            1, model.observation_space.shape[0]).astype(np.float32)
        ref, _ = model.predict(obs, deterministic=True)
        ts = torch.jit.load(str(tmp_path / "unpacked" / "policy_ts.pt")).eval()
        with torch.no_grad():
            out = ts(torch.from_numpy(obs)).numpy()
        assert np.abs(ref - out).max() < 1e-6
