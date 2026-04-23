"""scripts/phase2_smoke.py — Phase 2 verification.

Runs adapter.rollout() on the checkpoint produced by Phase 1 (trained
inline here, capped at 20,000 steps to honor the overnight-run policy)
with n_episodes=6, then confirms:
  - rollout.mp4 exists and size > 0
  - behavior.json is valid JSON
  - keyframes/ contains 12 PNGs (or fewer if episodes were shorter)
  - trajectory.npz is non-empty

Also exercises load_adapter() on examples/hopper/config.toml.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

HARD_CAP_STEPS = 20_000


def main() -> int:
    t0 = time.time()
    here = Path(__file__).resolve().parent.parent
    reward_module_path = here / "examples" / "hopper" / "rewards" / "v0.py"
    config_path = here / "examples" / "hopper" / "config.toml"

    from sculptor.adapters.base import load_adapter
    from sculptor.adapters.gym_sb3 import GymSB3Adapter

    # ── load_adapter roundtrip ──────────────────────────────────────────────
    adapter = load_adapter(config_path)
    assert isinstance(adapter, GymSB3Adapter), type(adapter)
    print(f"[phase2_smoke] load_adapter({config_path.name}) -> {type(adapter).__name__}")

    with tempfile.TemporaryDirectory(prefix="sculptor_p2_") as td:
        out = Path(td)
        train_dir = out / "train"
        rollout_dir = out / "rollout"

        # ── train (cap 20k) ────────────────────────────────────────────────
        steps = HARD_CAP_STEPS
        print(f"[phase2_smoke] training Hopper-v4 for {steps} steps -> {train_dir}")
        train_result = adapter.train(
            reward_module_path=reward_module_path,
            output_dir=train_dir,
            steps=steps,
            seed=42,
        )
        assert train_result.checkpoint_path.exists()

        # ── rollout (6 eps) ────────────────────────────────────────────────
        print(f"[phase2_smoke] rolling out 6 episodes -> {rollout_dir}")
        rollout_result = adapter.rollout(
            checkpoint_path=train_result.checkpoint_path,
            output_dir=rollout_dir,
            n_episodes=6,
        )

        # ── assertions ─────────────────────────────────────────────────────
        vp = rollout_result.video_path
        assert vp.exists(), f"rollout.mp4 missing: {vp}"
        assert vp.stat().st_size > 0, f"rollout.mp4 empty: {vp}"

        kdir = rollout_result.keyframes_dir
        pngs = sorted(kdir.glob("*.png"))
        assert kdir.is_dir(), f"keyframes dir missing: {kdir}"
        assert len(pngs) > 0, f"no PNGs in {kdir}"

        tp = rollout_result.trajectory_path
        assert tp.exists() and tp.stat().st_size > 0, f"trajectory.npz empty: {tp}"

        bp = rollout_dir / "behavior.json"
        assert bp.exists(), f"behavior.json missing: {bp}"
        behavior = json.loads(bp.read_text())
        assert "n_episodes" in behavior, behavior

        elapsed = time.time() - t0
        print("[phase2_smoke] PASSED")
        print(f"  elapsed_s:    {elapsed:.1f}")
        print(f"  video:        {vp} ({vp.stat().st_size} bytes)")
        print(f"  keyframes:    {len(pngs)} PNGs in {kdir}")
        print(f"  trajectory:   {tp} ({tp.stat().st_size} bytes)")
        print(f"  behavior:     {json.dumps(behavior, indent=2)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"[phase2_smoke] FAILED: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[phase2_smoke] FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
