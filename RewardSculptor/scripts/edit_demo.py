"""scripts/edit_demo.py — live verification of `sculptor.edit.apply_edits`.

Applies a hand-built Diagnosis (one literature-cited edit + one novel edit)
to `examples/hopper/rewards/v0.py`, calls the LLM, writes v1.py and the
re-exporting `current.py`, and prints the v1 REWARD_SPEC.references block.

Requires ANTHROPIC_API_KEY (auto-loaded from .env by `sculptor/__init__.py`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sculptor  # triggers .env load

from sculptor.adapters.base import load_adapter
from sculptor.diagnose import Diagnosis, ProposedEdit
from sculptor.edit import apply_edits


REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "examples" / "hopper" / "config.toml"
REWARDS_DIR = REPO / "examples" / "hopper" / "rewards"
V0 = REWARDS_DIR / "v0.py"
NEW_VERSION = "v1"

BEHAVIOR_GOAL = "run forward as fast as possible without falling"


# Hand-built diagnosis: one cited edit (DM Control — bounded alive bonus) +
# one novel edit (raise the control cost to a standard MuJoCo locomotion level).
DIAGNOSIS = Diagnosis(
    failure_modes=["premature_termination", "component_imbalance"],
    evidence=(
        "Every episode terminates by step ~42 with fall_rate=1.0. "
        "The alive_bonus of 1.0 per step plus forward_velocity ~0.68 "
        "gives enough return (~70) that the agent is rewarded for a short "
        "forward lunge before falling, and the tiny ctrl_cost (0.0015) "
        "provides no stabilizing pressure."
    ),
    proposed_edits=[
        ProposedEdit(
            target_term="alive_bonus",
            operation="increase",
            rationale=(
                "Raise the per-step alive bonus to 3.0 per DM Control's "
                "bounded-reward guidance — a single component should not "
                "dominate at the default weight; keeping the hopper upright "
                "must be at least as valuable as a short forward burst."
            ),
            suggested_value="3.0",
            paper_refs=["1801.00690"],
        ),
        ProposedEdit(
            target_term="ctrl_cost_weight",
            operation="increase",
            rationale=(
                "novel. The current ctrl_cost (~0.0015) exerts negligible "
                "stabilizing pressure, allowing jerky lunges that precipitate "
                "falls. Bump to 0.01 to penalize aggressive actuation that "
                "correlates with tipping — standard shaping choice in MuJoCo "
                "locomotion benchmarks, not literature-grounded here."
            ),
            suggested_value="0.01",
            paper_refs=[],
        ),
    ],
    literature_context=[],
    confidence=0.75,
    iter_dir=str(REPO / "runs" / "iter_000"),
    behavior_goal=BEHAVIOR_GOAL,
)


def main() -> int:
    adapter = load_adapter(CONFIG)
    contract = adapter.reward_contract()

    print(f"[edit_demo] applying {len(DIAGNOSIS.proposed_edits)} edits to "
          f"{V0} -> {REWARDS_DIR / (NEW_VERSION + '.py')}", flush=True)

    out_path = apply_edits(
        current_reward_path=V0,
        diagnosis=DIAGNOSIS,
        new_iter_id=NEW_VERSION,
        reward_contract=contract,
    )
    print(f"[edit_demo] wrote {out_path}", flush=True)

    # Load v1 fresh and print its REWARD_SPEC.references.
    import importlib.util

    spec = importlib.util.spec_from_file_location("v1_demo", out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rs = mod.REWARD_SPEC

    print()
    print("=" * 72)
    print(f"v1 REWARD_SPEC summary")
    print("=" * 72)
    print(f"version:     {rs['version']}")
    print(f"parent_hash: {rs['parent_hash']}")
    print(f"author:      {rs['author']}")
    print(f"description: {rs['description']}")
    print()
    print("hyperparameters:")
    for k, v in rs["hyperparameters"].items():
        print(f"  {k:<20} {v}")
    print()
    print("references:")
    print(json.dumps(rs["references"], indent=2, sort_keys=True))
    print()

    # Also show current.py actually works.
    cur_path = REWARDS_DIR / "current.py"
    cur_spec = importlib.util.spec_from_file_location("current_demo", cur_path)
    cur_mod = importlib.util.module_from_spec(cur_spec)
    cur_spec.loader.exec_module(cur_mod)
    print(f"current.py -> version {cur_mod.REWARD_SPEC['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
