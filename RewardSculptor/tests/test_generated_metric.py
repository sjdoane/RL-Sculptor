"""§Ship 35: generated-metric runtime (load/compute/resolve) + the
validation gates. GPU-free; metrics are written to temp .py files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.eval.generated_metric import (
    compute_generated_metric,
    make_generated_fitness_fn,
    resolve_fitness_fn,
)
from sculptor.eval.metric_validate import validate_generated_metric

# A valid task metric: forward travel × uprightness, physical + bounded.
GOOD = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); grav = arrays.get("projected_gravity_b")
    if root is None or grav is None:
        return {"spec_score": 0.0}
    disp = float(np.linalg.norm(root[-1, :, :2].mean(0) - root[0, :, :2].mean(0)))
    up = float(np.mean(grav[..., 2] < -0.85))
    speed = 1.0 - float(np.exp(-disp / 2.0))
    return {"spec_score": float(np.clip(speed * up, 0.0, 1.0)), "disp": disp, "up": up}
'''

BAD_IMPORT = '''import os
import numpy as np
def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.5}
'''

BAD_ARRAY = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    x = arrays["contact_forces"]
    return {"spec_score": float(np.clip(x.mean(), 0, 1))}
'''

REWARDS_STILLNESS = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel")
    motion = float(np.mean(np.abs(jv))) if jv is not None else 0.0
    return {"spec_score": float(max(0.0, 1.0 - motion))}
'''

NONDETERMINISTIC = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    return {"spec_score": float(np.random.random())}
'''

# §Ship 35 review: __builtins__ is in every module namespace without an
# import → a code-exec escape if the safety gate only blocks imports.
BUILTINS_ESCAPE = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    b = __builtins__
    return {"spec_score": 0.5}
'''

# §Ship 36: a metric that merely rewards motion MAGNITUDE — it scores a
# standing-and-flailing policy (the G1 kick hack) as highly as a competent
# one. The `upright_flail` archetype + non-degeneracy gate must reject it.
FLAIL_REWARDER = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    if jv is None or grav is None:
        return {"spec_score": 0.0}
    motion = float(np.mean(np.abs(jv)))
    up = float(np.mean(grav[..., 2] < -0.85))
    return {"spec_score": float(np.clip(motion / 5.0 * up, 0.0, 1.0))}
'''


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


def test_validate_good_metric_passes(tmp_path):
    p = _write(tmp_path, "good.py", GOOD)
    v = validate_generated_metric(GOOD, p)
    assert v["ok"], v["reasons"]
    assert all(v["gates"].values())
    # active archetype must outscore still/fallen.
    s = v["archetype_scores"]
    assert s["active"] > s["still"] and s["active"] > s["fallen"]


def test_validate_rejects_forbidden_import(tmp_path):
    p = _write(tmp_path, "bi.py", BAD_IMPORT)
    v = validate_generated_metric(BAD_IMPORT, p)
    assert not v["ok"]
    assert v["gates"]["ast_safety"] is False
    assert any("os" in r for r in v["reasons"])


def test_validate_rejects_builtins_escape(tmp_path):
    p = _write(tmp_path, "be.py", BUILTINS_ESCAPE)
    v = validate_generated_metric(BUILTINS_ESCAPE, p)
    assert not v["ok"]
    assert v["gates"]["ast_safety"] is False
    assert any("__builtins__" in r for r in v["reasons"])


def test_validate_rejects_unavailable_array(tmp_path):
    p = _write(tmp_path, "ba.py", BAD_ARRAY)
    v = validate_generated_metric(BAD_ARRAY, p)
    assert not v["ok"]
    assert v["gates"]["array_contract"] is False
    assert any("contact_forces" in r for r in v["reasons"])


def test_validate_rejects_rewarding_stillness(tmp_path):
    p = _write(tmp_path, "still.py", REWARDS_STILLNESS)
    v = validate_generated_metric(REWARDS_STILLNESS, p)
    assert not v["ok"]
    assert v["gates"]["nondegeneracy"] is False


def test_validate_rejects_flail_rewarder(tmp_path):
    """§Ship 36: a motion-magnitude metric that scores the stand-and-flail
    hack as high as competent motion fails the new upright_flail gate."""
    p = _write(tmp_path, "flail.py", FLAIL_REWARDER)
    v = validate_generated_metric(FLAIL_REWARDER, p)
    assert not v["ok"]
    assert v["gates"]["nondegeneracy"] is False
    assert any("upright_flail" in r for r in v["reasons"]), v["reasons"]
    assert "upright_flail" in v["archetype_scores"]


def test_validate_rejects_nondeterministic(tmp_path):
    p = _write(tmp_path, "nd.py", NONDETERMINISTIC)
    v = validate_generated_metric(NONDETERMINISTIC, p)
    assert not v["ok"]
    assert v["gates"]["determinism"] is False


def test_compute_and_resolve_generated_metric(tmp_path):
    p = _write(tmp_path, "good.py", GOOD)
    # synthetic rollout dir: forward-travelling, upright.
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    T, Ee = 50, 4
    root = np.zeros((T, Ee, 3), dtype=np.float32); root[..., 2] = 0.5
    root[..., 0] = (np.arange(T) * 0.05)[:, None]
    grav = np.zeros((T, Ee, 3), dtype=np.float32); grav[..., 2] = -1.0
    np.savez(rollout / "trajectory.npz", root_link_pos_w=root,
             projected_gravity_b=grav)
    (rollout / "behavior.json").write_text(json.dumps({"step_dt": 0.02}), encoding="utf-8")

    out = compute_generated_metric(p, rollout)
    assert 0.0 <= out["spec_score"] <= 1.0 and out["spec_score"] > 0.5
    # fitness fn scores iter_dir/rollout.
    fit = make_generated_fitness_fn(p)
    assert fit(tmp_path) == pytest.approx(out["spec_score"])
    # resolver dispatches a .py path to the generated fitness fn.
    assert resolve_fitness_fn(str(p))(tmp_path) == pytest.approx(out["spec_score"])
    # missing rollout → honest 0.0, never raises.
    assert make_generated_fitness_fn(p)(tmp_path / "nope") == 0.0


# ── generator (mock LLM) + calibration ────────────────────────────────

CONSTANT = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.5}
'''


class _Block:
    def __init__(self, text): self.type = "text"; self.text = text


class _Resp:
    def __init__(self, text): self.content = [_Block(text)]


class _Parsed:
    def __init__(self, parsed): self.parsed_output = parsed


class _Messages:
    def __init__(self, src, approved): self._src = src; self._approved = approved; self.creates = 0
    def create(self, **kw):
        self.creates += 1
        return _Resp("```python\n" + self._src + "\n```")
    def parse(self, **kw):
        from sculptor.eval.metric_gen import MetricReview
        return _Parsed(MetricReview(approved=self._approved,
                                    concerns=[] if self._approved else ["gameable"],
                                    summary="x"))


class _FakeClient:
    def __init__(self, src, approved=True): self.messages = _Messages(src, approved)


def test_generate_objective_metric_accepts_good(tmp_path):
    from sculptor.eval.metric_gen import generate_objective_metric
    out = tmp_path / "m"
    rec = generate_objective_metric(
        "trot forward", out, client=_FakeClient(GOOD, approved=True), max_attempts=2)
    assert rec["accepted"] and rec["validation_passed"]
    assert rec["review"]["approved"]
    assert (out / "metric.py").is_file() and (out / "meta.json").is_file()
    assert rec["calibrated"] is False  # steer-rights not granted yet


def test_generate_objective_metric_rejects_bad_and_retries(tmp_path):
    from sculptor.eval.metric_gen import generate_objective_metric
    client = _FakeClient(REWARDS_STILLNESS, approved=True)
    rec = generate_objective_metric(
        "trot forward", tmp_path / "m2", client=client, max_attempts=3)
    assert not rec["accepted"] and not rec["validation_passed"]
    assert client.messages.creates == 3  # retried up to the cap on failure


def test_generate_objective_metric_review_can_veto(tmp_path):
    from sculptor.eval.metric_gen import generate_objective_metric
    # validation passes but the reviewer rejects → not accepted.
    rec = generate_objective_metric(
        "trot forward", tmp_path / "m3",
        client=_FakeClient(GOOD, approved=False), max_attempts=1)
    assert rec["validation_passed"] and not rec["accepted"]


def test_calibration_good_metric_correlates(tmp_path):
    from sculptor.eval.metric_calibration import calibrate_metric
    p = _write(tmp_path, "good.py", GOOD)
    cal = calibrate_metric(p, "go1_trot", threshold=0.7)
    assert cal["ok"] and cal["spearman"] >= 0.7, cal


def test_calibration_constant_metric_fails(tmp_path):
    from sculptor.eval.metric_calibration import calibrate_metric
    p = _write(tmp_path, "const.py", CONSTANT)
    cal = calibrate_metric(p, "go1_trot", threshold=0.7)
    assert not cal["ok"]  # constant → no rank info → spearman 0
