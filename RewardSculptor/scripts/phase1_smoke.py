"""scripts/phase1_smoke.py — the Phase 1 verification gate's training smoke.

Trains Hopper-v4 for exactly 20,000 PPO timesteps with the v0 seed reward
and asserts the `TrainResult` shape Sculptor contracts require.

Run:
    uv run python scripts/phase1_smoke.py
Exits 0 on success, 1 on failure. Prints a concise summary.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

HARD_CAP_STEPS = 20_000  # per overnight-run spec, do not exceed


def main() -> int:
    from sculptor.adapters.gym_sb3 import GymSB3Adapter

    t0 = time.time()
    here = Path(__file__).resolve().parent.parent
    reward_module_path = here / "examples" / "hopper" / "rewards" / "v0.py"
    assert reward_module_path.exists(), reward_module_path

    # Match `examples/hopper/config.toml` exactly.
    adapter = GymSB3Adapter(
        env_id="Hopper-v4",
        n_envs=4,
        ppo_kwargs={"learning_rate": 3e-4, "n_steps": 2048},
    )

    steps = min(20_000, HARD_CAP_STEPS)
    with tempfile.TemporaryDirectory(prefix="sculptor_smoke_") as td:
        out = Path(td)
        print(f"[phase1_smoke] training Hopper-v4 for {steps} steps -> {out}")
        result = adapter.train(
            reward_module_path=reward_module_path,
            output_dir=out,
            steps=steps,
            seed=42,
        )

        # Gate assertions — see spec Phase 1 step 3
        assert result.checkpoint_path.exists(), result.checkpoint_path
        assert result.checkpoint_path.stat().st_size > 0, (
            f"empty checkpoint: {result.checkpoint_path}")
        assert "mean_return" in result.metrics_dict, result.metrics_dict
        assert isinstance(result.component_means, dict) and result.component_means, (
            "component_means is empty or not a dict")
        for k, v in result.component_means.items():
            assert isinstance(k, str) and isinstance(v, float), (k, type(v), v)
        assert result.logs_path.exists(), result.logs_path

        # Read back the metrics.json the adapter wrote.
        metrics_path = out / "metrics.json"
        assert metrics_path.exists(), metrics_path
        summary = json.loads(metrics_path.read_text())

        elapsed = time.time() - t0
        print("[phase1_smoke] PASSED")
        print(f"  steps:            {steps}")
        print(f"  elapsed_s:        {elapsed:.1f}")
        print(f"  mean_return:      {result.metrics_dict['mean_return']:+.3f}")
        print(f"  std_return:       {result.metrics_dict['std_return']:+.3f}")
        print(f"  component_means:  {json.dumps(result.component_means, indent=2)}")
        print(f"  checkpoint_size:  {result.checkpoint_path.stat().st_size} bytes")
        print(f"  metrics.json:     {metrics_path}")
        print(f"  metrics summary:  {json.dumps(summary, indent=2)}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"[phase1_smoke] FAILED: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[phase1_smoke] FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
