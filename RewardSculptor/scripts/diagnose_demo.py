"""scripts/diagnose_demo.py — end-to-end Hopper v0 verification of
`sculptor.diagnose.diagnose()`.

What this does:
  1. Train Hopper-v4 with examples/hopper/rewards/v0.py for 20,000 steps
     (the overnight-mode hard cap) into runs/iter_000/.
  2. Roll out 6 eval episodes into the same dir.
  3. Run the two-stage diagnoser against that directory with behavior goal
     "run forward as fast as possible without falling".
  4. Pretty-print the Diagnosis.

Re-running will re-use the existing iter dir if train/rollout artifacts
are already present (`--reuse`). Otherwise everything is produced fresh.

Run:
    uv run python scripts/diagnose_demo.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import sculptor  # triggers .env auto-load

from sculptor.adapters.base import load_adapter
from sculptor.diagnose import diagnose, print_diagnosis


REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "examples" / "hopper" / "config.toml"
REWARD_V0 = REPO / "examples" / "hopper" / "rewards" / "v0.py"
BEHAVIOR_GOAL = "run forward as fast as possible without falling"

HARD_CAP_STEPS = 20_000


def _iter_dir_ready(d: Path) -> bool:
    need = [
        d / "checkpoint.zip",
        d / "metrics.json",
        d / "reward_spec.json",
        d / "behavior.json",
        d / "keyframes",
        d / "trajectory.npz",
    ]
    return all(p.exists() for p in need) and any((d / "keyframes").glob("*.png"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", type=Path,
                    default=REPO / "runs" / "iter_000")
    ap.add_argument("--reuse", action="store_true",
                    help="Skip train+rollout if iter_dir already has all artifacts.")
    ap.add_argument("--skip-rollout", action="store_true",
                    help="Re-use an existing checkpoint; run only the rollout + diagnose.")
    ap.add_argument("--n-episodes", type=int, default=6)
    args = ap.parse_args()

    iter_dir: Path = args.iter_dir
    iter_dir.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(CONFIG)

    if args.reuse and _iter_dir_ready(iter_dir):
        print(f"[demo] reusing iter_dir at {iter_dir}", flush=True)
    else:
        t0 = time.time()
        print(f"[demo] training Hopper-v4 for {HARD_CAP_STEPS} steps -> {iter_dir}",
              flush=True)
        adapter.train(
            reward_module_path=REWARD_V0,
            output_dir=iter_dir,
            steps=HARD_CAP_STEPS,
            seed=42,
        )
        print(f"[demo]   train done in {time.time() - t0:.1f}s", flush=True)

        if not args.skip_rollout:
            t1 = time.time()
            print(f"[demo] rolling out {args.n_episodes} episodes "
                  f"-> {iter_dir}", flush=True)
            adapter.rollout(
                checkpoint_path=iter_dir / "checkpoint.zip",
                output_dir=iter_dir,
                n_episodes=args.n_episodes,
            )
            print(f"[demo]   rollout done in {time.time() - t1:.1f}s", flush=True)

    if not _iter_dir_ready(iter_dir):
        print(f"[demo] iter_dir incomplete at {iter_dir}; aborting", file=sys.stderr)
        return 1

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[demo] ANTHROPIC_API_KEY not set — diagnose() will fail. "
              "Check .env or shell env.", file=sys.stderr)
        return 2

    print(f"[demo] diagnosing with goal: {BEHAVIOR_GOAL!r}", flush=True)
    t2 = time.time()
    d = diagnose(iter_dir, BEHAVIOR_GOAL, CONFIG)
    print(f"[demo]   diagnose done in {time.time() - t2:.1f}s", flush=True)
    print()
    print_diagnosis(d)
    print()
    print(f"[demo] wrote {iter_dir / 'diagnosis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
