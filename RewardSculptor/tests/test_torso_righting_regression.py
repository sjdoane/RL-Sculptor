"""D23 golden regression: the live sat-up rollout that scored fitness 0.0.

On 2026-07-12 (g1-standing / starting-from-lying-flat-on-its, torso_righting
iter 0) the policy sat up correctly from the settled supine reset — end-window
projected-gravity z −0.917 (more upright than the reference clip's own end
window), joints articulating — and the certified stage metric scored 0.0,
because it was certified against the FULL lying→standing clip and demanded
z_end ≥ 0.35 / rise ≥ 0.20, which a physically correct floor-sit can never
satisfy (pelvis stays ~0.14 m). See REFERENCE_BUILD_LOG.md D23.

The fixture directory snapshots the live artifacts:
  - trajectory.npz        rollout arrays, 500 frames x 16 envs (sliced from 64)
  - behavior.json         rollout behavior meta (step_dt etc.)
  - metric_as_shipped.py  the certified metric that produced the live 0.0
  - meta_as_shipped.json  its certification record
  - reference_clip.npz    the attached clip (fallandgetup2_subject3--seg00)
  - eval_reset.json       the stage's settled eval reset

These tests PIN the failure class: perfect goal behavior, zero score. The
companion tests for the FIXED pipeline (sub-span certification, D24) assert
the repaired metric class scores this same rollout well above zero.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "torso_righting_satup"


def _load_metric(path: Path):
    spec = importlib.util.spec_from_file_location(f"fixture_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def satup_arrays() -> dict:
    with np.load(FIXTURE_DIR / "trajectory.npz") as npz:
        return {k: npz[k] for k in npz.files}


@pytest.fixture(scope="module")
def satup_behavior() -> dict:
    return json.loads((FIXTURE_DIR / "behavior.json").read_text())


def test_shipped_metric_zeroes_the_correct_situp(satup_arrays, satup_behavior):
    """The as-shipped full-clip-certified metric scores the correct sit-up
    at exactly 0.0 — the D23 failure, pinned so the class stays visible."""
    mod = _load_metric(FIXTURE_DIR / "metric_as_shipped.py")
    out = mod.compute_spec(satup_arrays, satup_behavior, {})
    assert out["spec_score"] == 0.0
    # ... while every non-height channel confirms the behavior was RIGHT:
    assert out["gate_upright_frac"] == 1.0
    assert out["gate_started_low_frac"] == 1.0
    assert out["gate_joint_motion_frac"] == 1.0
    assert out["gravity_z_end_mean"] < -0.85  # torso fully righted
    # ... and the zero came from full-clip height demands alone:
    assert out["gate_reached_035_frac"] == 0.0
    assert out["gate_rise_floor_frac"] == 0.0
    assert out["z_end_mean"] < 0.20  # physical floor-sit pelvis height


def test_fixture_rollout_is_a_genuine_situp(satup_arrays):
    """Sanity on the snapshot itself: starts low and supine-ish, ends low
    with the torso upright, joints moving — the exact shape the stage goal
    asks for. Guards against a silently corrupted/re-exported fixture."""
    z = satup_arrays["root_link_pos_w"][:, :, 2]
    g_z = satup_arrays["projected_gravity_b"][:, :, 2]
    assert z.shape[0] == 500 and z.shape[1] >= 8
    assert float(np.mean(z[:25])) < 0.25          # starts on the ground
    assert float(np.mean(z[-35:])) < 0.25         # stays on the ground (sit)
    # The policy rights the torso within ~0.5 s of the reset (g_z −0.19 at
    # frame 0, −0.84 by frame 25) and then HOLDS for the remaining 9.5 s —
    # the completion-then-hold shape the F3 gate exists for. Only the very
    # first frames still show the settled supine start.
    assert float(np.mean(g_z[:5])) > -0.60        # starts non-upright
    assert float(np.mean(g_z[-35:])) < -0.85      # ends torso-upright
    jv = np.mean(np.abs(satup_arrays["joint_vel"]))
    assert 0.05 < float(jv) < 3.0                 # real articulation


def test_fixed_metric_scores_the_situp_well_above_zero(
    satup_arrays, satup_behavior,
):
    """The repaired torso_righting metric (certified against the
    goal-aligned 0-8.1 s sub-span; live regen 2026-07-12, all six
    reference gates green) scores the same rollout 1.0.

    Honest scope (Opus audit L1): this pins the snapshotted ARTIFACT
    (metric_fixed.py), not the pipeline — it would still pass if the
    span/validation code regressed. The pipeline behavior itself is
    pinned by test_reference_spans.py (selection + QC),
    test_reference_anchored_validation.py (the six gates and their
    exploit families), and test_criterion_ground.py (criterion
    re-grounding); together with this file the class is covered from
    both ends."""
    fixed = FIXTURE_DIR / "metric_fixed.py"
    if not fixed.is_file():
        pytest.skip("metric_fixed.py not yet snapshotted (D24 in flight)")
    mod = _load_metric(fixed)
    out = mod.compute_spec(satup_arrays, satup_behavior, {})
    assert out["spec_score"] >= 0.5
