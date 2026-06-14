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
