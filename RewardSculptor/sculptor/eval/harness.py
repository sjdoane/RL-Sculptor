"""The eval harness (§Ship 27 / E2): seeds × conditions × benchmarks.

`run_campaign` executes every (benchmark, condition, seed) job
sequentially (one GPU), each in its own scaffolded mini-project under
the campaign dir, computes the benchmark's spec metric on EVERY
iteration's rollout, and aggregates per (benchmark, condition) with
IQM + stratified bootstrap CIs. Seeds are PAIRED across conditions
(same seed list), which is what makes small-n comparisons honest.

Remote dispatch needs zero harness plumbing: export SCULPTOR_REMOTE_*
(or run through the UI's settings) and every adapter.train call routes
through the Ship-23 executor automatically.

Resumability: a job whose `result.json` exists is skipped — campaigns
survive crashes, pod restarts, and Ctrl-C without redoing GPU work.
Failed jobs are recorded as honest zeros with their error (silently
dropping failures inflates aggregates).
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import sqlite3
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from sculptor.eval.benchmarks import (
    BENCHMARKS,
    BenchmarkTask,
    benchmark_registry,
    get_benchmark,
)
from sculptor.eval.charter import (
    freeze_or_verify_campaign_charter,
    verify_result_lineage,
)
from sculptor.eval.spec_metrics import compute_spec_metrics, make_spec_fitness_fn
from sculptor.eval.stats import iqm, stratified_bootstrap_ci

#: Adapter short-name → dotted class path for scaffolded configs.
_ADAPTER_CLASSES = {
    "mjlab": "sculptor.adapters.mjlab.MjlabAdapter",
}

#: The universal starter reward shared by every condition that begins
#: from a seed reward — byte-identical across conditions by
#: construction (fairness: nobody gets a better v0). Same template
#: `sculpt init` scaffolds for batched adapters.
_V0_TEMPLATE = '''"""v0 — shared starter reward for eval campaigns (constant alive bonus)."""

from __future__ import annotations


REWARD_SPEC: dict = {
    "version": "v0",
    "parent_hash": "",
    "description": "Eval-campaign starter: constant alive bonus per env.",
    "author": "human",
    "supports_batched": True,
    "hyperparameters": {"alive_bonus": 1.0},
    "references": [],
}


def compute_reward(state, action, next_state, info):
    alive = float(REWARD_SPEC["hyperparameters"]["alive_bonus"])
    return alive, {"alive_bonus": alive}


def compute_reward_batched(state, action, next_state, info):
    import torch

    n = action.shape[0]
    alive = float(REWARD_SPEC["hyperparameters"]["alive_bonus"])
    rewards = torch.full((n,), alive, device=action.device, dtype=action.dtype)
    return rewards, {"alive_bonus": rewards.clone()}
'''


@dataclass(frozen=True)
class EvalCondition:
    """One experimental condition. `mode`:
    * "mission"    — the FULL SYSTEM: LLM curriculum decomposition →
                     per-stage sculpt loops (no_kg toggles the KG
                     ablation across decompose/diagnose/edit).
    * "sculpt"     — single-stage sculpt loop (the plan's
                     NO-CURRICULUM ablation; no_kg toggles KG).
    * "train_only" — one training run, no LLM anywhere:
                     use_seed_reward=False → the env's intrinsic reward
                     (plain-PPO baseline); True → the shared v0 starter
                     (the plan's NO-DIAGNOSE ablation: seed, never
                     iterate — use the *_matched variant for equal GPU).

    E4 condition mapping (plan §E4 → harness names):
      full system    → mission
      no-KG          → mission_no_kg
      no-curriculum  → full
      no-diagnose    → seed_only_matched
      baselines (E3) → plain_ppo(+_matched), eureka

    COMPUTE FAIRNESS: each sculpt-loop iteration trains FROM SCRATCH
    for steps_per_iter, so a sculpt job consumes `iterations ×
    steps_per_iter` rsl_rl iterations of GPU while a plain train_only
    job consumes `1 × steps_per_iter`. `compute_matched=True` scales a
    train_only condition's single run to the SAME total budget — use
    the `*_matched` variants when the comparison must hold GPU
    constant rather than iteration-loop count. Every result records
    `total_rl_iterations` so the report can never hide the difference.
    """

    name: str
    mode: str
    no_kg: bool = True
    use_seed_reward: bool = True
    compute_matched: bool = False
    notes: str = ""


CONDITIONS: dict[str, EvalCondition] = {
    c.name: c
    for c in (
        EvalCondition(
            name="mission", mode="mission", no_kg=False,
            notes="FULL SYSTEM: KG-grounded curriculum decomposition + "
                  "per-stage diagnose/edit loops",
        ),
        EvalCondition(
            name="mission_no_kg", mode="mission", no_kg=True,
            notes="E4 no-KG ablation: identical mission flow, KG "
                  "stripped from decompose/diagnose/edit",
        ),
        EvalCondition(
            name="full", mode="sculpt", no_kg=False,
            notes="E4 no-curriculum ablation: single-stage KG-grounded "
                  "diagnose -> edit loop (no decomposition)",
        ),
        EvalCondition(
            name="no_kg", mode="sculpt", no_kg=True,
            notes="single-stage loop, KG stripped (no-curriculum AND "
                  "no-KG)",
        ),
        EvalCondition(
            name="plain_ppo", mode="train_only", use_seed_reward=False,
            notes="baseline: env intrinsic reward, no LLM, 1x steps",
        ),
        EvalCondition(
            name="plain_ppo_matched", mode="train_only",
            use_seed_reward=False, compute_matched=True,
            notes="baseline: intrinsic reward at the sculpt jobs' TOTAL "
                  "GPU budget (iterations x steps)",
        ),
        EvalCondition(
            name="seed_only", mode="train_only", use_seed_reward=True,
            notes="ablation: shared v0 starter, no diagnose/edit, 1x steps",
        ),
        EvalCondition(
            name="seed_only_matched", mode="train_only",
            use_seed_reward=True, compute_matched=True,
            notes="ablation: v0 starter at the sculpt jobs' TOTAL GPU budget",
        ),
        EvalCondition(
            name="eureka", mode="eureka",
            notes="E3 baseline: K LLM reward candidates per generation, "
                  "select by spec fitness, reward reflection — no KG, no "
                  "diagnosis, no curriculum (see eval/eureka.py deltas; "
                  "trains generations x K runs)",
        ),
    )
}


@dataclass
class CampaignConfig:
    name: str
    out_dir: Path
    benchmarks: list[str]
    conditions: list[str]
    seeds: list[int]
    #: Optional strict suite fragments. They can add capability-described
    #: tasks but cannot override built-ins.
    benchmark_manifests: list[Path] = field(default_factory=list)
    #: sculpt-mode LLM-loop iterations per job.
    iterations: int = 2
    #: rsl_rl iterations per training run.
    steps_per_iter: int = 300
    rollout_episodes: int = 4
    #: spec_score threshold for iterations-to-criterion.
    spec_threshold: float = 0.5
    #: Eureka condition: candidates per generation (generations =
    #: `iterations`, so a eureka job trains iterations × eureka_k runs).
    eureka_k: int = 4
    #: §Ship 33: feed the benchmark spec metric INTO the sculpt loop as a
    #: ground-truth fitness signal (fitness-guided diagnose + best-by-
    #: fitness selection + plateau early-stop). Removes the eval asymmetry
    #: where only eureka could select on fitness. Default False = the
    #: blind, spec-unaware loop (the original E4 condition).
    fitness_in_loop: bool = False
    #: test seam — overrides the adapter class in scaffolded configs.
    adapter_class_override: Optional[str] = None

    def validate(
        self, registry: Optional[dict[str, BenchmarkTask]] = None,
    ) -> None:
        registry = BENCHMARKS if registry is None else registry
        if not self.benchmarks:
            raise ValueError("campaign needs at least one benchmark")
        if len(set(self.benchmarks)) != len(self.benchmarks):
            raise ValueError("duplicate benchmarks would duplicate campaign jobs")
        for b in self.benchmarks:
            benchmark = get_benchmark(b, registry)
            if not benchmark.campaign_ready:
                limitations = "; ".join(benchmark.known_limitations)
                raise ValueError(
                    f"benchmark {b!r} is {benchmark.evaluation_tier} and not "
                    f"campaign-ready: {limitations}"
                )
            if benchmark.spec_metric is None:
                raise ValueError(
                    f"benchmark {b!r} has no objective spec metric"
                )
            if (
                benchmark.adapter not in _ADAPTER_CLASSES
                and self.adapter_class_override is None
            ):
                raise ValueError(
                    f"benchmark {b!r} uses adapter {benchmark.adapter!r} but "
                    "the eval harness has no adapter class mapping for it"
                )
        if not self.conditions:
            raise ValueError("campaign needs at least one condition")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("duplicate conditions would duplicate campaign jobs")
        for c in self.conditions:
            if c not in CONDITIONS:
                raise KeyError(
                    f"unknown condition {c!r}; known: {sorted(CONDITIONS)}"
                )
        if not self.seeds:
            raise ValueError("campaign needs at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate seeds — pairing requires distinct seeds")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.eureka_k < 1:
            raise ValueError("eureka_k must be >= 1")


def _emit(payload: dict[str, Any]) -> None:
    print("[SCULPT-EVENT] " + json.dumps(payload, default=str), flush=True)


def _job_dir(cfg: CampaignConfig, bench: str, cond: str, seed: int) -> Path:
    return Path(cfg.out_dir) / bench / cond / f"seed_{seed}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _campaign_uses_kg(cfg: CampaignConfig) -> bool:
    return any(
        CONDITIONS[name].mode in {"sculpt", "mission"}
        and not CONDITIONS[name].no_kg
        for name in cfg.conditions
    )


def _prepare_campaign_kg_snapshot(
    cfg: CampaignConfig,
) -> tuple[Optional[Path], Optional[dict[str, Any]]]:
    """Freeze one consistent KG input for all KG-enabled campaign jobs.

    Each job later receives its own writable copy.  This prevents a run case
    learned by an earlier seed/condition from contaminating a later paired
    arm, while still allowing crash-resume inside one job.
    """
    if not _campaign_uses_kg(cfg):
        return None, None
    from sculptor.eval.charter import CHARTER_FILENAME, CharterIntegrityError
    from sculptor.kg.store import SculptorKG, default_db_path
    from sculptor.run_context import write_json_atomic

    inputs_dir = Path(cfg.out_dir) / "campaign_inputs"
    snapshot = inputs_dir / "kg_base.db"
    record_path = inputs_dir / "kg_base.json"
    charter_exists = (Path(cfg.out_dir) / CHARTER_FILENAME).is_file()
    if snapshot.exists() or record_path.exists():
        if not snapshot.is_file() or not record_path.is_file():
            raise CharterIntegrityError(
                "campaign KG snapshot is incomplete (expected kg_base.db and "
                "kg_base.json)"
            )
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CharterIntegrityError(
                f"campaign KG snapshot record is unreadable: {exc}"
            ) from exc
        actual = _file_sha256(snapshot)
        if record.get("sha256") != actual:
            raise CharterIntegrityError(
                "campaign KG base snapshot hash mismatch; the frozen input "
                "was altered"
            )
        return snapshot, {
            "kind": "sqlite_kg_snapshot",
            "sha256": actual,
            "size_bytes": snapshot.stat().st_size,
        }
    if charter_exists:
        raise CharterIntegrityError(
            "campaign charter exists but its KG base snapshot is missing"
        )

    inputs_dir.mkdir(parents=True, exist_ok=True)
    source = default_db_path()
    tmp = inputs_dir / "kg_base.db.tmp"
    try:
        if source.is_file():
            # SQLite's backup API includes committed WAL state and produces a
            # transactionally consistent snapshot even if the source is open.
            src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
            dst = sqlite3.connect(str(tmp))
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
        else:
            # An absent shared KG is a legitimate, explicitly pinned empty KG.
            with SculptorKG(tmp):
                pass
        os.replace(tmp, snapshot)
    finally:
        if tmp.exists():
            tmp.unlink()
    snapshot_hash = _file_sha256(snapshot)
    record = {
        "schema_version": 1,
        "source_path": str(source),
        "source_existed": source.is_file(),
        "sha256": snapshot_hash,
        "size_bytes": snapshot.stat().st_size,
    }
    write_json_atomic(record_path, record)
    return snapshot, {
        "kind": "sqlite_kg_snapshot",
        "sha256": snapshot_hash,
        "size_bytes": snapshot.stat().st_size,
    }


def _prepare_job_kg(
    job_dir: Path, base_snapshot: Path, base_sha256: str,
) -> Path:
    inputs = job_dir / "inputs"
    job_kg = inputs / "kg.db"
    origin_path = inputs / "kg_origin.json"
    if job_kg.exists() or origin_path.exists():
        if not job_kg.is_file() or not origin_path.is_file():
            raise RuntimeError(
                f"incomplete per-job KG state under {inputs}; refusing an "
                "ambiguous resume"
            )
        origin = json.loads(origin_path.read_text(encoding="utf-8"))
        if origin.get("base_sha256") != base_sha256:
            raise RuntimeError(
                f"per-job KG lineage does not match campaign snapshot: {job_kg}"
            )
        return job_kg
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_snapshot, job_kg)
    from sculptor.run_context import write_json_atomic

    write_json_atomic(origin_path, {
        "schema_version": 1,
        "base_sha256": base_sha256,
        "policy": "private_writable_copy_per_campaign_job",
    })
    return job_kg


def _scaffold_job_project(
    cfg: CampaignConfig, bench: BenchmarkTask, seed: int, job_dir: Path,
) -> Path:
    """Minimal sculpt project: config.toml + rewards/v0.py. No git —
    sculpt_run's commit helpers no-op outside a repo."""
    (job_dir / "rewards").mkdir(parents=True, exist_ok=True)
    (job_dir / "runs").mkdir(exist_ok=True)
    (job_dir / "reports").mkdir(exist_ok=True)
    (job_dir / "rewards" / "__init__.py").write_text("", encoding="utf-8")
    v0 = job_dir / "rewards" / "v0.py"
    if not v0.exists():
        v0.write_text(_V0_TEMPLATE, encoding="utf-8")

    adapter_class = cfg.adapter_class_override or _ADAPTER_CLASSES[bench.adapter]
    adapter_cfg_items = {"task_id": bench.task_id, **bench.adapter_config}
    cfg_inline = ", ".join(
        f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {json.dumps(v)}"
        for k, v in adapter_cfg_items.items()
    )
    config = f'''[target]
name = "{cfg.name}-{bench.name}-{seed}"

[adapter]
class = "{adapter_class}"
config = {{ {cfg_inline} }}

[kg]
environment_tag = "{bench.name}"

[iteration]
steps_per_iter = {cfg.steps_per_iter}
primary_metric = "mean_return"
behavior_metrics = []
rollout_episodes = {cfg.rollout_episodes}
seed = {seed}
auto_adjust_physics = false
early_stop_enabled = false
'''
    (job_dir / "config.toml").write_text(config, encoding="utf-8")
    return job_dir / "config.toml"


def _run_mission_mode(
    cfg: CampaignConfig,
    bench: BenchmarkTask,
    condition: EvalCondition,
    seed: int,
    job_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    """The full-system condition: decompose the goal into a mission
    once (resume reuses the existing decomposition — the curriculum is
    part of the seed's experiment state), then mission_run with the
    campaign's per-stage budget. Returns {n_stages, completed,
    halted_reason} for the result record."""
    from sculptor.adapters.base import load_adapter
    from sculptor.cli import _derive_mission_slug
    from sculptor.decompose import decompose_task
    from sculptor.mission import load_mission, save_mission
    from sculptor.sculpt import mission_run

    missions_root = job_dir / ".missions"
    missions_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in missions_root.iterdir() if p.is_dir())

    def _open_kg():
        if condition.no_kg:
            return None
        from sculptor.kg.store import SculptorKG

        return SculptorKG()

    if existing:
        mission = load_mission(missions_root / existing[0])
    else:
        adapter = load_adapter(config_path)
        kg_store = _open_kg()
        try:
            mission = decompose_task(
                bench.behavior_goal,
                adapter.reward_contract(),
                kg_store=kg_store,
                skill_library_handle=None,
            )
        finally:
            if kg_store is not None:
                kg_store.close()
        slug = _derive_mission_slug(bench.behavior_goal, set())
        mission.mission_dir = str((missions_root / slug).resolve())
        save_mission(mission, missions_root / slug)

    kg_store = _open_kg()
    try:
        result = mission_run(
            mission,
            adapter_short_name=bench.adapter,
            kg_store=kg_store,
            iterations_override=cfg.iterations,
            steps_per_iter=cfg.steps_per_iter,
            seed=seed,
            # Adaptive early stop is part of the SYSTEM under test —
            # a stage exits the moment its criterion holds.
            early_stop_on_criterion=True,
            # §Ship 34: when fitness-in-loop is on, every stage's sculpt
            # loop is guided by (and best-selects on) the benchmark spec
            # metric — uniform across stages (sound for single-skill
            # benchmarks; the curriculum sub-goals all serve the one task).
            fitness_metric=(bench.spec_metric if cfg.fitness_in_loop else None),
        )
    finally:
        if kg_store is not None:
            kg_store.close()
    return {
        "n_stages": len(mission.stages),
        "completed": bool(getattr(result, "completed", False)),
        "halted_reason": getattr(result, "halted_reason", None),
    }


def _mission_iterations_used(job_dir: Path) -> Optional[int]:
    """Sum of per-stage iterations_used from the persisted mission.json
    — survives crashes (saved after every stage transition), which is
    what makes the GPU accounting honest in failure paths."""
    missions_root = job_dir / ".missions"
    if not missions_root.is_dir():
        return None
    for md in sorted(missions_root.iterdir()):
        mj = md / "mission.json"
        if mj.is_file():
            try:
                doc = json.loads(mj.read_text(encoding="utf-8"))
                return sum(
                    int(s.get("iterations_used", 0) or 0)
                    for s in doc.get("stages", [])
                )
            except Exception:  # noqa: BLE001
                return None
    return None


def _mission_spec_series(
    cfg: CampaignConfig, bench: BenchmarkTask, job_dir: Path,
) -> list[dict[str, Any]]:
    """Spec metric across the mission's stages in curriculum order,
    indexed by a job-global iteration counter (the comparison axis is
    total LLM-loop iterations spent, regardless of which stage spent
    them). Each entry keeps its (stage, stage_iter) provenance."""
    out: list[dict[str, Any]] = []
    missions_root = job_dir / ".missions"
    if not missions_root.is_dir():
        return out
    idx = 0
    for md in sorted(missions_root.iterdir()):
        mj = md / "mission.json"
        if not mj.is_file():
            continue
        try:
            doc = json.loads(mj.read_text(encoding="utf-8"))
            stage_names = [s["name"] for s in doc.get("stages", [])]
        except Exception:  # noqa: BLE001
            stage_names = []
        for stage_name in stage_names:
            runs = md / "stages" / stage_name / "runs"
            if not runs.is_dir():
                continue
            iters = sorted(
                (d for d in runs.glob("iter_*") if d.is_dir()),
                key=lambda d: int(d.name.split("_", 1)[1]),
            )
            for it in iters:
                result = compute_spec_metrics(bench.spec_metric, it / "rollout")
                result["iter"] = idx
                result["stage"] = stage_name
                result["stage_iter"] = int(it.name.split("_", 1)[1])
                out.append(result)
                idx += 1
    return out


def _spec_series(
    cfg: CampaignConfig, bench: BenchmarkTask, job_dir: Path,
) -> list[dict[str, Any]]:
    """Spec metric for every completed iteration, in iter order."""
    out: list[dict[str, Any]] = []
    runs = job_dir / "runs"
    iters = sorted(
        (d for d in runs.glob("iter_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_", 1)[1]),
    )
    for it in iters:
        rollout = it / "rollout"
        result = compute_spec_metrics(bench.spec_metric, rollout)
        result["iter"] = int(it.name.split("_", 1)[1])
        out.append(result)
    return out


def _run_job(
    cfg: CampaignConfig,
    bench: BenchmarkTask,
    condition: EvalCondition,
    seed: int,
    charter_design_sha256: str,
    kg_base_snapshot: Optional[Path] = None,
    kg_base_sha256: Optional[str] = None,
) -> dict[str, Any]:
    job_dir = _job_dir(cfg, bench.name, condition.name, seed)
    result_path = job_dir / "result.json"
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("charter_design_sha256") != charter_design_sha256:
            from sculptor.eval.charter import CharterIntegrityError

            raise CharterIntegrityError(
                f"cached result has no matching charter lineage: {result_path}"
            )
        cached["cached"] = True
        return cached

    config_path = _scaffold_job_project(cfg, bench, seed, job_dir)
    t0 = time.monotonic()
    error: Optional[str] = None
    eureka_summary: Optional[dict[str, Any]] = None
    mission_summary: Optional[dict[str, Any]] = None
    uses_kg = condition.mode in {"sculpt", "mission"} and not condition.no_kg
    old_rs_kg = os.environ.get("RS_KG_PATH")
    had_rs_kg = "RS_KG_PATH" in os.environ
    if uses_kg:
        if kg_base_snapshot is None or kg_base_sha256 is None:
            raise RuntimeError("KG-enabled eval job has no frozen campaign KG input")
        job_kg = _prepare_job_kg(job_dir, kg_base_snapshot, kg_base_sha256)
        os.environ["RS_KG_PATH"] = str(job_kg)
    try:
        if condition.mode == "train_only":
            from sculptor.adapters.base import load_adapter

            adapter = load_adapter(config_path)
            iter_dir = job_dir / "runs" / "iter_0"
            iter_dir.mkdir(parents=True, exist_ok=True)
            reward = (
                (job_dir / "rewards" / "v0.py").resolve()
                if condition.use_seed_reward else None
            )
            steps = cfg.steps_per_iter * (
                cfg.iterations if condition.compute_matched else 1
            )
            train_res = adapter.train(
                reward_module_path=reward,
                output_dir=iter_dir,
                steps=steps,
                seed=seed,
            )
            adapter.rollout(
                checkpoint_path=train_res.checkpoint_path,
                output_dir=iter_dir / "rollout",
                n_episodes=cfg.rollout_episodes,
            )
        elif condition.mode == "sculpt":
            from sculptor.sculpt import sculpt_run

            # §Ship 33: optional fitness-in-loop — the loop sees the
            # benchmark's spec metric as ground-truth fitness (same signal
            # eureka selects on), making diagnose/selection fitness-guided.
            fitness_fn = (
                make_spec_fitness_fn(bench.spec_metric)
                if cfg.fitness_in_loop else None
            )
            sculpt_run(
                config_path=config_path,
                behavior_goal=bench.behavior_goal,
                iterations=cfg.iterations,
                resume=True,           # idempotent across harness restarts
                no_kg=condition.no_kg,
                steps_per_iter=cfg.steps_per_iter,
                rollout_episodes=cfg.rollout_episodes,
                seed=seed,
                fitness_fn=fitness_fn,
                fitness_target=cfg.spec_threshold,
            )
        elif condition.mode == "eureka":
            from sculptor.eval.eureka import run_eureka_job

            eureka_summary = run_eureka_job(
                config_path=config_path,
                job_dir=job_dir,
                behavior_goal=bench.behavior_goal,
                spec_metric=bench.spec_metric,
                generations=cfg.iterations,
                k=cfg.eureka_k,
                steps_per_iter=cfg.steps_per_iter,
                rollout_episodes=cfg.rollout_episodes,
                seed=seed,
            )
        elif condition.mode == "mission":
            mission_summary = _run_mission_mode(
                cfg, bench, condition, seed, job_dir, config_path,
            )
        else:  # pragma: no cover — guarded by validate()
            raise ValueError(f"unknown condition mode {condition.mode!r}")
    except Exception as e:  # noqa: BLE001 — honest zero, campaign continues
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        if uses_kg:
            if had_rs_kg:
                assert old_rs_kg is not None
                os.environ["RS_KG_PATH"] = old_rs_kg
            else:
                os.environ.pop("RS_KG_PATH", None)

    wall = time.monotonic() - t0
    if condition.mode == "mission":
        series = _mission_spec_series(cfg, bench, job_dir)
    else:
        series = _spec_series(cfg, bench, job_dir)
    scores = [s.get("spec_score", 0.0) for s in series]
    best = max(scores) if scores else 0.0
    if condition.mode == "eureka":
        # Eureka's DEFINED output (Ma et al. Algorithm 1) is the best
        # reward across all generations — scoring its last generation
        # would strawman the baseline. Sculpt conditions are scored on
        # their last reward because they cannot select on the spec...
        final = best
    elif condition.mode in ("sculpt", "mission") and cfg.fitness_in_loop:
        # §Ship 33: ...UNLESS fitness-in-loop is on — then the loop DID
        # select on the spec (best-by-fitness current.py), so its defined
        # output is the best, scored apples-to-apples with eureka.
        final = best
    else:
        final = scores[-1] if scores else 0.0
    # Threshold accounting uses the REAL iter index (a generation whose
    # candidates all failed leaves a hole in the series; enumerate
    # position would silently shift everything after it).
    iters_to = next(
        (int(s["iter"]) + 1 for s in series
         if s.get("spec_score", 0.0) >= cfg.spec_threshold),
        None,
    )
    captures = [s.get("capture") for s in series if s.get("capture")]
    if condition.mode == "train_only":
        total_rl_iters = cfg.steps_per_iter * (
            cfg.iterations if condition.compute_matched else 1
        )
    elif condition.mode == "mission":
        used = _mission_iterations_used(job_dir)
        total_rl_iters = cfg.steps_per_iter * (
            used if used is not None else len(series)
        )
    elif condition.mode == "eureka":
        if eureka_summary is not None:
            trained = eureka_summary.get("candidates_trained", 0)
        else:
            # The job crashed before returning a summary — recover the
            # GPU accounting from the per-generation log flush, else
            # from trained-candidate dirs on disk. Billing zero for
            # generations that demonstrably trained would corrupt the
            # fairness comparison exactly in the failure path.
            try:
                _log = json.loads(
                    (job_dir / "eureka_log.json").read_text(encoding="utf-8")
                )
                trained = int(_log.get("candidates_trained", 0))
            except Exception:  # noqa: BLE001
                trained = len(list(
                    (job_dir / "eureka").glob("gen_*/cand_*/train/checkpoint.pt")
                ))
        total_rl_iters = cfg.steps_per_iter * trained
    else:
        total_rl_iters = cfg.steps_per_iter * len(series)
    result: dict[str, Any] = {
        "charter_design_sha256": charter_design_sha256,
        "benchmark": bench.name,
        "condition": condition.name,
        "seed": seed,
        "final_spec_score": float(final),
        # Each condition is scored by its method's DEFINED output:
        # eureka selects on fitness (Algorithm 1 returns the best
        # across generations); sculpt conditions cannot select on the
        # spec and are scored on their last reward.
        "final_rule": (
            "best_across_generations" if condition.mode == "eureka"
            else "best_across_iterations"
            if condition.mode in ("sculpt", "mission") and cfg.fitness_in_loop
            else "last_iteration"
        ),
        "best_spec_score": float(best),
        "iters_to_threshold": iters_to,
        "spec_threshold": cfg.spec_threshold,
        "iterations_completed": len(series),
        # The GPU-budget number comparisons must be read against.
        "total_rl_iterations": int(total_rl_iters),
        "wall_seconds": round(wall, 1),
        "spec_series": series,
        "capture": captures[-1] if captures else None,
        "eureka": eureka_summary,
        "mission": mission_summary,
        "error": error,
    }
    from sculptor.run_context import write_json_atomic

    write_json_atomic(result_path, result)
    return result


def aggregate(results: list[dict[str, Any]], *, rng_seed: int = 0) -> dict[str, Any]:
    """Per-(benchmark, condition) aggregates over PAIRED seeds + a
    capture-parity audit (frame-domain spec metrics are comparable only
    under identical capture settings — E1 persists them precisely so
    this can be checked instead of assumed)."""
    by_bc: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in results:
        by_bc.setdefault((r["benchmark"], r["condition"]), []).append(r)

    agg: dict[str, Any] = {}
    parity_warnings: list[str] = []
    for (bench, cond), rs in sorted(by_bc.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        finals = [r["final_spec_score"] for r in rs]
        bests = [r["best_spec_score"] for r in rs]
        entry = {
            "seeds": [r["seed"] for r in rs],
            "final_spec_scores": finals,
            "final": stratified_bootstrap_ci(finals, rng_seed=rng_seed),
            "best": stratified_bootstrap_ci(bests, rng_seed=rng_seed),
            "iters_to_threshold": [r["iters_to_threshold"] for r in rs],
            "threshold_hit_rate": (
                sum(1 for r in rs if r["iters_to_threshold"] is not None)
                / len(rs)
            ),
            "total_rl_iterations": sorted(
                {r.get("total_rl_iterations") for r in rs} - {None}
            ),
            "mean_wall_seconds": round(
                sum(r["wall_seconds"] for r in rs) / len(rs), 1,
            ),
            "errors": [r["error"] for r in rs if r.get("error")],
        }
        agg.setdefault(bench, {})[cond] = entry

    # Paired per-seed DIFFERENCES — the statistic condition comparisons
    # must be made on (seeds are paired by design; comparing two
    # independent CIs overlapping is weaker and over-conservative).
    pairwise: dict[str, dict[str, Any]] = {}
    for bench, conds in agg.items():
        names = sorted(conds)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ra = {s: v for s, v in zip(conds[a]["seeds"],
                                           conds[a]["final_spec_scores"])}
                rb = {s: v for s, v in zip(conds[b]["seeds"],
                                           conds[b]["final_spec_scores"])}
                shared = sorted(set(ra) & set(rb))
                if not shared:
                    continue
                diffs = [ra[s] - rb[s] for s in shared]
                pairwise.setdefault(bench, {})[f"{a} - {b}"] = {
                    "seeds": shared,
                    "diffs": diffs,
                    "diff": stratified_bootstrap_ci(diffs, rng_seed=rng_seed),
                }

    # Capture parity within each benchmark, across ALL its conditions.
    # A job with NO capture info counts as its own bucket — mixed
    # missing/real captures must warn too, not hide behind the None.
    by_bench: dict[str, set[str]] = {}
    for r in results:
        cap = r.get("capture")
        by_bench.setdefault(r["benchmark"], set()).add(
            json.dumps(cap, sort_keys=True) if cap else "(no capture info)"
        )
    for bench, caps in sorted(by_bench.items()):
        if len(caps) > 1:
            parity_warnings.append(
                f"benchmark {bench!r} mixes capture settings: "
                + " vs ".join(sorted(caps))
            )
    return {
        "benchmarks": agg,
        "pairwise": pairwise,
        "capture_parity_warnings": parity_warnings,
    }


def _report_html(campaign: dict[str, Any]) -> str:
    """Self-contained single-file report: per-benchmark bars (IQM of
    final spec) with bootstrap-CI whiskers. No JS, no external deps."""
    rows: list[str] = []
    for bench, conds in sorted(campaign["aggregates"]["benchmarks"].items()):
        bars: list[str] = []
        h = 24 + len(conds) * 30 + 8
        for i, (cond, e) in enumerate(sorted(conds.items())):
            x = 140
            y = 24 + i * 30
            w = 360
            point, lo, hi = (
                e["final"]["point"], e["final"]["ci_low"], e["final"]["ci_high"],
            )
            bars.append(
                f'<text x="8" y="{y + 13}" font-size="12">{cond}</text>'
                f'<rect x="{x}" y="{y}" width="{max(1, point * w):.0f}" '
                f'height="16" fill="#5fd0a0" />'
                f'<line x1="{x + lo * w:.0f}" x2="{x + hi * w:.0f}" '
                f'y1="{y + 8}" y2="{y + 8}" stroke="#333" stroke-width="2" />'
                f'<text x="{x + w + 12}" y="{y + 13}" font-size="12">'
                f'{point:.3f} [{lo:.3f}, {hi:.3f}] n={e["final"]["n"]}</text>'
            )
        rows.append(
            f"<h3>{bench}</h3>"
            f'<svg width="680" height="{h}" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(bars)}</svg>'
        )
    pair_rows: list[str] = []
    for bench, pairs in sorted(campaign["aggregates"].get("pairwise", {}).items()):
        for label, p in sorted(pairs.items()):
            d = p["diff"]
            sig = (
                "≠ 0" if (d["ci_low"] > 0 or d["ci_high"] < 0)
                else "CI spans 0"
            )
            pair_rows.append(
                f"<tr><td>{bench}</td><td>{label}</td>"
                f"<td>{d['point']:+.3f}</td>"
                f"<td>[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]</td>"
                f"<td>{sig}</td><td>n={d['n']}</td></tr>"
            )
    pairwise_html = (
        "<h2>Paired per-seed differences (final spec)</h2>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
        "<tr><th>benchmark</th><th>comparison</th><th>IQM Δ</th>"
        "<th>95% CI</th><th></th><th></th></tr>"
        + "".join(pair_rows) + "</table>"
    ) if pair_rows else ""
    all_warnings = [
        *campaign["aggregates"]["capture_parity_warnings"],
        *campaign.get("authority_warnings", []),
    ]
    warn = "".join(
        f'<p style="color:#b00">⚠ {w}</p>'
        for w in all_warnings
    )
    coverage = campaign.get("coverage")
    coverage_html = ""
    if isinstance(coverage, dict):
        complete = bool(coverage.get("complete"))
        state = "COMPLETE" if complete else "INCOMPLETE"
        missing = int(coverage.get("n_missing", 0))
        expected = int(coverage.get("n_expected", 0))
        completed = int(coverage.get("n_completed", 0))
        color = "#176b3a" if complete else "#9a3412"
        coverage_html = (
            f"<p style='border:1px solid {color};padding:.75rem;color:{color}'>"
            f"<strong>Coverage {state}</strong>: {completed}/{expected} jobs; "
            f"{missing} missing. Partial aggregates are not a complete "
            "campaign result.</p>"
        )
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>eval: {campaign['name']}</title>"
        "<body style='font-family:system-ui;max-width:760px;margin:2rem auto'>"
        f"<h1>{campaign['name']}</h1>"
        f"<p>{campaign['created_at']} — final spec_score IQM with 95% "
        "stratified-bootstrap CIs over paired seeds. Compare conditions "
        "via the paired-difference table, and read every comparison "
        "against its <code>total_rl_iterations</code> GPU budget.</p>"
        f"{coverage_html}{warn}{''.join(rows)}{pairwise_html}</body>"
    )


def run_campaign(
    cfg: CampaignConfig,
    *,
    job_hook: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Execute the campaign (resumable) and write
    `<out_dir>/campaign_report.json` + `report.html`. Returns the
    report dict."""
    registry = benchmark_registry(cfg.benchmark_manifests)
    cfg.validate(registry)
    authority_warnings = [
        f"benchmark {name!r} uses spec {registry[name].spec_metric!r} with "
        f"authority {registry[name].spec_authority!r}, not a verified "
        "A4_reporting certificate; treat its campaign result as provisional"
        for name in cfg.benchmarks
        if registry[name].spec_authority != "A4_reporting"
    ]
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kg_base_snapshot, kg_input = _prepare_campaign_kg_snapshot(cfg)
    external_inputs: dict[str, Any] = {}
    if kg_input is not None:
        external_inputs["knowledge_graph"] = kg_input
    if cfg.benchmark_manifests:
        external_inputs["benchmark_manifests"] = [
            {
                "sha256": _file_sha256(Path(path).expanduser().resolve()),
                "size_bytes": Path(path).expanduser().resolve().stat().st_size,
            }
            for path in cfg.benchmark_manifests
        ]
    # Freeze the experiment before emitting a start event or doing any GPU/LLM
    # work.  Resumes must match the entire design and executable source tree.
    charter = freeze_or_verify_campaign_charter(
        cfg,
        benchmarks=registry,
        conditions=CONDITIONS,
        external_inputs=external_inputs,
    )
    charter_hash = str(charter["design_sha256"])
    jobs = [
        (b, c, s)
        for b in cfg.benchmarks
        for c in cfg.conditions
        for s in cfg.seeds
    ]
    _emit({
        "type": "eval_campaign_started",
        "campaign": cfg.name,
        "n_jobs": len(jobs),
        "benchmarks": cfg.benchmarks,
        "conditions": cfg.conditions,
        "seeds": cfg.seeds,
        "charter_design_sha256": charter_hash,
        "authority_warnings": authority_warnings,
    })
    results: list[dict[str, Any]] = []
    for i, (b, c, s) in enumerate(jobs):
        bench = get_benchmark(b, registry)
        condition = CONDITIONS[c]
        _emit({
            "type": "eval_job_started",
            "job": f"{b}/{c}/seed_{s}",
            "index": i + 1,
            "total": len(jobs),
        })
        r = _run_job(
            cfg,
            bench,
            condition,
            s,
            charter_hash,
            kg_base_snapshot,
            kg_input["sha256"] if kg_input is not None else None,
        )
        results.append(r)
        _emit({
            "type": "eval_job_finished",
            "job": f"{b}/{c}/seed_{s}",
            "cached": bool(r.get("cached")),
            "final_spec_score": r["final_spec_score"],
            "iters_to_threshold": r["iters_to_threshold"],
            "wall_seconds": r["wall_seconds"],
            "error": r.get("error"),
        })
        if job_hook is not None:
            job_hook(r)

    from datetime import datetime, timezone

    from sculptor.run_context import capture_run_context, write_json_atomic

    verify_result_lineage(results, charter)
    report = {
        "name": cfg.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "charter": {
            "path": str(out / "campaign_charter.json"),
            "schema_version": charter["schema_version"],
            "design_sha256": charter_hash,
            "document_sha256": charter["document_sha256"],
        },
        "campaign_inputs": external_inputs,
        "authority_warnings": authority_warnings,
        "config": {
            **{k: v for k, v in asdict(cfg).items() if k != "out_dir"},
            "out_dir": str(cfg.out_dir),
        },
        "aggregates": aggregate(results),
        "jobs": [
            {k: v for k, v in r.items() if k != "spec_series"}
            for r in results
        ],
        # R1: the campaign itself is an experiment — pin its context.
        "run_context": capture_run_context(out, None, behavior_goal=cfg.name),
    }
    write_json_atomic(out / "campaign_report.json", report)
    (out / "report.html").write_text(_report_html(report), encoding="utf-8")
    _emit({
        "type": "eval_campaign_finished",
        "campaign": cfg.name,
        "report": str(out / "campaign_report.json"),
        "parity_warnings": report["aggregates"]["capture_parity_warnings"],
    })
    return report
