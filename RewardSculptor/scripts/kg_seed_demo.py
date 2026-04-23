"""scripts/kg_seed_demo.py — hand-seed the KG with realistic entities
derived from the already-ingested seed papers, then demo query_techniques /
query_semantic / cite against them.

WHY THIS EXISTS
---------------
The live extraction path (`sculpt kg extract --all`) calls Claude Opus 4.7
and requires `ANTHROPIC_API_KEY`. Claude Code's OAuth token exposed to this
process is rejected by the public API (confirmed: HTTP 401 "OAuth
authentication is currently not supported"). So this script stands in for
one invocation of `extract_entities` per paper, using hand-written
`ExtractionPayload`s that mirror each paper's actual content.

When you run `sculpt kg extract --all` with a real API key, the payloads
will be LLM-generated but the schema, materialization, and query surface
are identical — this script exercises the same _materialize codepath that
extract_entities uses in production.

Run:
    uv run python scripts/kg_seed_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Local imports — must follow venv activation via `uv run`
from sculptor.kg.extract import (
    EnvironmentExtract,
    ExtractionPayload,
    FailureModeExtract,
    PaperToEnvironmentRel,
    PaperToTechniqueRel,
    RewardComponentExtract,
    TechniqueExtract,
    TechniqueToFailureModeRel,
    TechniqueToRewardComponentRel,
    _materialize,
)
from sculptor.kg.query import cite, query_semantic, query_techniques
from sculptor.kg.schema import Paper
from sculptor.kg.store import SculptorKG


# ── Hand-written payloads ──────────────────────────────────────────────────
# Each one is what a careful reader would produce from the paper's own text,
# structured to the same ExtractionPayload Pydantic schema the LLM uses.

_PPO = ExtractionPayload(
    techniques=[
        TechniqueExtract(
            name="proximal_policy_optimization",
            description=(
                "Policy gradient algorithm that alternates between sampling data "
                "and optimizing a clipped surrogate objective. Simpler than TRPO "
                "while matching or beating its performance."
            ),
            tags=["on_policy", "policy_gradient", "actor_critic"],
            evidence=(
                "We propose a new family of policy gradient methods for "
                "reinforcement learning, which alternate between sampling data "
                "through interaction with the environment, and optimizing a "
                "'surrogate' objective function using stochastic gradient ascent."
            ),
        ),
        TechniqueExtract(
            name="clipped_surrogate_objective",
            description=(
                "Ratio-clipping on the importance-sampled policy update, "
                "preventing excessively large policy steps without a trust-"
                "region constraint."
            ),
            tags=["regularization", "stability"],
            evidence=(
                "We propose a novel objective with clipped probability ratios, "
                "which forms a pessimistic estimate (i.e., lower bound) of the "
                "performance of the policy."
            ),
        ),
    ],
    failure_modes=[
        FailureModeExtract(
            name="policy_collapse",
            description=(
                "Trust-region-free policy gradient methods destabilize when "
                "updates are too large, collapsing learned behavior."
            ),
            symptoms=["sudden drop in return", "KL spike between updates"],
            environment_tag="continuous_locomotion",
            evidence=(
                "Standard policy gradient methods perform one gradient update "
                "per data sample, and have a hard time getting good results; "
                "they can suffer from destructively large policy updates."
            ),
        ),
    ],
    reward_components=[
        RewardComponentExtract(
            name="advantage_estimate",
            description="Generalized advantage estimator used in the surrogate.",
            formula="A_hat_t = sum_{l=0}^{T-t-1} (gamma*lambda)^l * delta_{t+l}",
            hyperparameters={"gamma": 0.99, "lambda": 0.95},
            evidence=(
                "When using neural networks that share parameters between the "
                "policy and value function, we must use a loss function that "
                "combines the policy surrogate and a value function error term; "
                "we use a truncated version of generalized advantage estimation."
            ),
        ),
    ],
    environments=[
        EnvironmentExtract(
            name="MuJoCo",
            description="Continuous-control robotic tasks from the MuJoCo physics engine.",
            tags=["continuous_locomotion", "mujoco"],
            evidence="We compared several different surrogate objectives... "
            "using the Mujoco physics simulator.",
        ),
    ],
    paper_to_technique=[
        PaperToTechniqueRel(technique="proximal_policy_optimization"),
        PaperToTechniqueRel(technique="clipped_surrogate_objective"),
    ],
    technique_to_failure_mode=[
        TechniqueToFailureModeRel(
            technique="clipped_surrogate_objective", failure_mode="policy_collapse"),
    ],
    technique_to_reward_component=[
        TechniqueToRewardComponentRel(
            technique="proximal_policy_optimization", reward_component="advantage_estimate"),
    ],
    paper_to_environment=[PaperToEnvironmentRel(environment="MuJoCo")],
)


_GYM = ExtractionPayload(
    techniques=[
        TechniqueExtract(
            name="standardized_env_api",
            description=(
                "A consistent reset/step/render interface for RL environments, "
                "enabling algorithm benchmarking across diverse tasks."
            ),
            tags=["benchmark", "infrastructure"],
            evidence=(
                "OpenAI Gym... a toolkit for reinforcement learning research. "
                "It includes a growing collection of benchmark problems that "
                "expose a common interface."
            ),
        ),
    ],
    environments=[
        EnvironmentExtract(
            name="Atari",
            description="Atari 2600 arcade games via the ALE.",
            tags=["discrete_control", "pixels"],
            evidence="Atari 2600 games, via the Arcade Learning Environment.",
        ),
        EnvironmentExtract(
            name="Classic_Control",
            description="Small classic control tasks (cartpole, mountain-car, etc.).",
            tags=["continuous_locomotion", "classic"],
            evidence="Classic control: the small-scale classic control problems.",
        ),
        EnvironmentExtract(
            name="MuJoCo_Gym",
            description="MuJoCo-based continuous control envs exposed via Gym (Hopper, "
            "HalfCheetah, Ant, Humanoid).",
            tags=["continuous_locomotion", "mujoco"],
            evidence="MuJoCo: continuous control tasks running in the MuJoCo simulator.",
        ),
    ],
    paper_to_technique=[PaperToTechniqueRel(technique="standardized_env_api")],
    paper_to_environment=[
        PaperToEnvironmentRel(environment="Atari"),
        PaperToEnvironmentRel(environment="Classic_Control"),
        PaperToEnvironmentRel(environment="MuJoCo_Gym"),
    ],
)


_DEEP_RL_THAT_MATTERS = ExtractionPayload(
    techniques=[
        TechniqueExtract(
            name="multi_seed_evaluation",
            description=(
                "Reporting RL results over many random seeds and taking "
                "confidence intervals to avoid single-run cherry-picking."
            ),
            tags=["evaluation", "reproducibility"],
            evidence=(
                "We show that commonly used methods to compare deep reinforcement "
                "learning algorithms can be highly sensitive to random seeds; "
                "we investigate how variance in results impacts reproducibility."
            ),
        ),
    ],
    failure_modes=[
        FailureModeExtract(
            name="reward_hacking",
            description=(
                "Agent exploits flaws in a hand-crafted reward rather than solving "
                "the intended task."
            ),
            symptoms=["high reward with low task success", "physics exploits"],
            environment_tag="continuous_locomotion",
            evidence=(
                "Choice of reward scale, function form, and hyper-parameters can "
                "cause algorithms to perform well for the wrong reasons, "
                "particularly when rewards are engineered."
            ),
        ),
        FailureModeExtract(
            name="evaluation_variance",
            description=(
                "Results across random seeds span a wide range, making headline "
                "numbers unreliable without repeated runs."
            ),
            symptoms=["wide seed-to-seed confidence intervals"],
            environment_tag="continuous_locomotion",
            evidence="We show that commonly used methods can be highly sensitive "
            "to random seeds... variance in results impacts reproducibility.",
        ),
        FailureModeExtract(
            name="sparse_reward",
            description=(
                "Task terminates or provides uninformative reward until rare "
                "success states, stalling learning."
            ),
            symptoms=["flat training curve", "no reward signal until end of episode"],
            environment_tag="continuous_locomotion",
            evidence="Reward shaping is especially important for sparse reward "
            "scenarios in continuous control tasks.",
        ),
        FailureModeExtract(
            name="hyperparameter_sensitivity",
            description=(
                "Minor changes in hyperparameters cause large changes in final "
                "performance, frustrating comparisons between algorithms."
            ),
            symptoms=["identical code, different conclusions"],
            environment_tag="continuous_locomotion",
            evidence="Hyperparameters can have significantly different effects "
            "across algorithms and environments.",
        ),
    ],
    paper_to_technique=[PaperToTechniqueRel(technique="multi_seed_evaluation")],
    technique_to_failure_mode=[
        TechniqueToFailureModeRel(
            technique="multi_seed_evaluation", failure_mode="evaluation_variance"),
        TechniqueToFailureModeRel(
            technique="multi_seed_evaluation", failure_mode="hyperparameter_sensitivity"),
        # The paper also argues multi-seed eval surfaces reward-hacking patterns
        # that single-seed cherry-picking would hide.
        TechniqueToFailureModeRel(
            technique="multi_seed_evaluation", failure_mode="reward_hacking"),
    ],
    paper_to_environment=[PaperToEnvironmentRel(environment="MuJoCo_Gym")],
)


_DM_CONTROL = ExtractionPayload(
    techniques=[
        TechniqueExtract(
            name="bounded_unit_reward",
            description=(
                "All DM Control tasks report rewards in [0, 1] per step so that "
                "returns are directly comparable across tasks."
            ),
            tags=["benchmark", "evaluation", "reward_shaping"],
            evidence=(
                "Each task yields a reward in [0, 1]. The reward is usually "
                "structured as a product or weighted sum of sigmoid or "
                "tolerance-based reward components."
            ),
        ),
        TechniqueExtract(
            name="tolerance_reward_kernel",
            description=(
                "Reward components built from smooth tolerance functions that "
                "saturate inside a desired interval and decay smoothly outside."
            ),
            tags=["reward_shaping", "dense_reward"],
            evidence=(
                "Individual reward components are typically smooth tolerance "
                "functions, which return 1.0 for values in a bounded interval "
                "and decay smoothly to 0 outside."
            ),
        ),
    ],
    failure_modes=[
        FailureModeExtract(
            name="reward_scale_inconsistency",
            description=(
                "Different tasks using different reward scales make algorithm "
                "comparisons across tasks noisy or meaningless."
            ),
            symptoms=["returns incomparable across tasks"],
            environment_tag="continuous_locomotion",
            evidence=(
                "Standardised reward structure permits algorithms to be evaluated "
                "consistently across tasks."
            ),
        ),
    ],
    reward_components=[
        RewardComponentExtract(
            name="tolerance_kernel",
            description="Smooth tolerance-based reward in [0, 1].",
            formula="r = 1 if x in [lo, hi] else sigmoid((hi-x)/m) * sigmoid((x-lo)/m)",
            evidence="tolerance() returns 1.0 for values in a bounded interval.",
        ),
    ],
    environments=[
        EnvironmentExtract(
            name="DeepMind_Control_Suite",
            description=(
                "A set of continuous-control tasks with a standardized reward "
                "structure, based on the MuJoCo physics engine."
            ),
            tags=["continuous_locomotion", "mujoco", "benchmark"],
            evidence=(
                "The DeepMind Control Suite is a set of continuous control tasks "
                "with a standardised structure and interpretable rewards."
            ),
        ),
    ],
    paper_to_technique=[
        PaperToTechniqueRel(technique="bounded_unit_reward"),
        PaperToTechniqueRel(technique="tolerance_reward_kernel"),
    ],
    technique_to_failure_mode=[
        TechniqueToFailureModeRel(
            technique="bounded_unit_reward", failure_mode="reward_scale_inconsistency"),
        # Bounding rewards to [0,1] narrows the gamable surface; tolerance
        # kernels provide dense signal that eliminates many sparse-reward
        # training plateaus.
        TechniqueToFailureModeRel(
            technique="bounded_unit_reward", failure_mode="reward_hacking"),
        TechniqueToFailureModeRel(
            technique="tolerance_reward_kernel", failure_mode="sparse_reward"),
    ],
    technique_to_reward_component=[
        TechniqueToRewardComponentRel(
            technique="tolerance_reward_kernel", reward_component="tolerance_kernel"),
    ],
    paper_to_environment=[PaperToEnvironmentRel(environment="DeepMind_Control_Suite")],
)


_PAYLOAD_BY_ARXIV_ID = {
    "1707.06347": _PPO,
    "1606.01540": _GYM,
    "1709.06560": _DEEP_RL_THAT_MATTERS,
    "1801.00690": _DM_CONTROL,
}


# For the two papers whose arxiv API requests hit HTTP 429 during initial
# ingest (so their `authors` ended up empty), backfill a canonical author list
# so cite() prints a real surname instead of "Unknown". A production ingest
# with the API live would populate these automatically — this stand-in keeps
# the demo citations readable without re-hitting a rate-limited endpoint.
_AUTHORS_BACKFILL = {
    "1707.06347": ["John Schulman", "Filip Wolski", "Prafulla Dhariwal",
                   "Alec Radford", "Oleg Klimov"],
    "1709.06560": ["Peter Henderson", "Riashat Islam", "Philip Bachman",
                   "Joelle Pineau", "Doina Precup", "David Meger"],
}


def _backfill_authors(store: SculptorKG) -> None:
    for arxiv_id, authors in _AUTHORS_BACKFILL.items():
        p = store.get_node(f"paper:{arxiv_id}")
        if isinstance(p, Paper) and not p.authors:
            p.authors = list(authors)
            if arxiv_id == "1707.06347":
                p.year = p.year or 2017
            if arxiv_id == "1709.06560":
                p.year = p.year or 2017
            store.add_node(p, upsert=True)


def seed_kg(store: SculptorKG) -> dict:
    """Run hand-written payloads through the same _materialize path extract uses."""
    summary: dict[str, int] = {}
    for arxiv_id, payload in _PAYLOAD_BY_ARXIV_ID.items():
        paper = store.get_node(f"paper:{arxiv_id}")
        if not isinstance(paper, Paper):
            print(f"[seed]   skip {arxiv_id} — not in store (run `sculpt kg ingest` first)",
                  file=sys.stderr)
            continue
        nodes, edges = _materialize(store, paper, payload)
        paper.extracted = True
        store.add_node(paper, upsert=True)
        summary[arxiv_id] = len(nodes)
        print(f"[seed]   {arxiv_id} -> {len(nodes)} nodes, {edges} edges")
    return summary


def run_demo():
    store = SculptorKG()
    try:
        print("=" * 72)
        print("STEP 0: backfill author lists rate-limited out of arXiv metadata")
        print("=" * 72)
        _backfill_authors(store)

        print()
        print("=" * 72)
        print("STEP 1: seed the KG with hand-written extraction payloads")
        print("(the same codepath `sculpt kg extract --all` uses with a real API key)")
        print("=" * 72)
        seed_kg(store)

        print()
        s = store.stats()
        print(f"[stats] nodes: {s['total_nodes']}  edges: {s['total_edges']}  "
              f"embeddings: {s['total_embeddings']}")
        print(f"[stats] by_kind: {s['nodes_by_kind']}")
        print(f"[stats] by_relation: {s['edges_by_relation']}")

        print()
        print("=" * 72)
        print("STEP 2: sample extraction payload (PPO)")
        print("=" * 72)
        print(json.dumps(_PPO.model_dump(), indent=2, sort_keys=True))

        print()
        print("=" * 72)
        print("STEP 3: query_techniques(['sparse_reward', 'reward_hacking'])")
        print("=" * 72)
        results = query_techniques(
            ["sparse_reward", "reward_hacking"], store=store, top_k=5)
        _print_matches(results)

        print()
        print("=" * 72)
        print("STEP 3b: query_techniques with domain_filter='continuous_locomotion'")
        print("=" * 72)
        filtered = query_techniques(
            ["sparse_reward", "reward_hacking"],
            domain_filter="continuous_locomotion",
            store=store, top_k=5)
        _print_matches(filtered)

        print()
        print("=" * 72)
        print("STEP 4: query_semantic('stabilize forward locomotion without "
              "exploiting simulator')")
        print("=" * 72)
        sem = query_semantic(
            "stabilize forward locomotion without exploiting simulator",
            top_k=5, store=store)
        _print_matches(sem, show_score=True)

        print()
        print("=" * 72)
        print("STEP 5: cite() against live-ingested papers")
        print("=" * 72)
        for arxiv_id in ("1707.06347", "1606.01540", "1709.06560", "1801.00690"):
            print(f"  {cite(arxiv_id, store=store)}")
    finally:
        store.close()


def _print_matches(matches, *, show_score: bool = False):
    if not matches:
        print("  (no matches)")
        return
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m.technique.name}")
        if show_score:
            print(f"     score:     {m.relevance_score:+.4f}")
        else:
            print(f"     score:     {m.relevance_score:.2f}  "
                  f"matched on {m.matched_on}")
        desc = m.description.strip()
        if len(desc) > 120:
            desc = desc[:120] + "..."
        print(f"     desc:      {desc}")
        print(f"     citation:  {m.paper_citation}")
        if m.evidence:
            ev = m.evidence.strip()
            if len(ev) > 140:
                ev = ev[:140] + "..."
            print(f"     evidence:  {ev}")


if __name__ == "__main__":
    run_demo()
