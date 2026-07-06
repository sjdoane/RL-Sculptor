"""§Ship 35: ground-texture spec_fn — verifies the MjSpec texture→
material→geom assignment COMPILES (offline; no GPU/render needed), the
env-var toggle, and chaining of any existing spec_fn. The visual result
still needs a GPU rollout to confirm — this guards the model-construction
API and the never-break-rollout contract."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

mujoco = pytest.importorskip("mujoco")

from sculptor.adapters import _mjlab_runner  # noqa: E402

_FLOOR_XML = (
    '<mujoco><worldbody>'
    '<geom name="{name}" type="{type}" size="2 2 0.1"/>'
    '</worldbody></mujoco>'
)


def test_ground_texture_assigns_and_compiles():
    scene = SimpleNamespace(spec_fn=None)
    env_cfg = SimpleNamespace(scene=scene)
    _mjlab_runner._apply_ground_texture(env_cfg)
    assert callable(scene.spec_fn), "spec_fn should be installed"
    spec = mujoco.MjSpec.from_string(_FLOOR_XML.format(name="terrain_0", type="box"))
    scene.spec_fn(spec)
    model = spec.compile()                       # must not raise
    assert model.ntex >= 1 and model.nmat >= 1
    assert model.geom_matid[0] >= 0              # terrain geom got a material


def test_ground_texture_matches_plane_floor():
    scene = SimpleNamespace(spec_fn=None)
    _mjlab_runner._apply_ground_texture(SimpleNamespace(scene=scene))
    spec = mujoco.MjSpec.from_string(_FLOOR_XML.format(name="floor", type="plane"))
    scene.spec_fn(spec)
    model = spec.compile()
    assert model.geom_matid[0] >= 0              # plane floor matched too


def test_ground_texture_disabled_by_env(monkeypatch):
    monkeypatch.setenv("SCULPTOR_GROUND_TEXTURE", "off")
    scene = SimpleNamespace(spec_fn=None)
    _mjlab_runner._apply_ground_texture(SimpleNamespace(scene=scene))
    assert scene.spec_fn is None                 # disabled → untouched


def test_ground_texture_chains_existing_spec_fn():
    calls = {"n": 0}

    def prev(_spec):
        calls["n"] += 1

    scene = SimpleNamespace(spec_fn=prev)
    _mjlab_runner._apply_ground_texture(SimpleNamespace(scene=scene))
    spec = mujoco.MjSpec.from_string(_FLOOR_XML.format(name="floor", type="plane"))
    scene.spec_fn(spec)
    spec.compile()
    assert calls["n"] == 1                        # existing spec_fn preserved


def test_ground_texture_no_scene_is_safe():
    # An env_cfg without a spec_fn-capable scene must not raise.
    _mjlab_runner._apply_ground_texture(SimpleNamespace(scene=SimpleNamespace()))
    _mjlab_runner._apply_ground_texture(SimpleNamespace())


# ── rollout viewer config (720p default + no ghost neighbor envs) ─────────

def _viewer_ns(**over):
    base = dict(width=320, height=240, max_extra_envs=2)
    base.update(over)
    return SimpleNamespace(**base)


def test_rollout_viewer_defaults_to_720p_no_extra_envs():
    viewer = _viewer_ns()
    _mjlab_runner._configure_rollout_viewer(
        SimpleNamespace(viewer=viewer), SimpleNamespace())
    assert viewer.width == 1280
    assert viewer.height == 720
    # The glitchy-background fix: neighbor envs auto-reset mid-episode,
    # so they must never be in frame.
    assert viewer.max_extra_envs == 0


def test_rollout_viewer_args_override():
    viewer = _viewer_ns()
    args = SimpleNamespace(render_width=960, render_height=540)
    _mjlab_runner._configure_rollout_viewer(SimpleNamespace(viewer=viewer), args)
    assert (viewer.width, viewer.height) == (960, 540)


def test_rollout_viewer_zero_args_mean_default():
    viewer = _viewer_ns()
    args = SimpleNamespace(render_width=0, render_height=0)
    _mjlab_runner._configure_rollout_viewer(SimpleNamespace(viewer=viewer), args)
    assert (viewer.width, viewer.height) == (1280, 720)


def test_rollout_viewer_clamps_tiny_sizes():
    viewer = _viewer_ns()
    args = SimpleNamespace(render_width=8, render_height=8)
    _mjlab_runner._configure_rollout_viewer(SimpleNamespace(viewer=viewer), args)
    assert viewer.width >= 64 and viewer.height >= 64


def test_rollout_viewer_missing_attrs_are_safe():
    # No viewer at all, and a viewer lacking the attributes: never raises.
    _mjlab_runner._configure_rollout_viewer(SimpleNamespace(), SimpleNamespace())
    bare = SimpleNamespace()
    _mjlab_runner._configure_rollout_viewer(
        SimpleNamespace(viewer=bare), SimpleNamespace())
    assert not hasattr(bare, "width")  # hasattr-guarded, not force-set
