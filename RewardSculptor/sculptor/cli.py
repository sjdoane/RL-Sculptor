"""sculptor/cli.py — `sculpt` command-line entry point.

Phase 1 scaffolding for top-level commands (`init`, `run`, `resume`, `viz`) —
they print "not implemented yet" until the inner loop lands. The `kg`
subcommand group (list-papers, list-techniques, stats) is wired to the real
store (see `sculptor.kg.store.SculptorKG`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import typer

from sculptor.kg.schema import NODE_TYPES, Paper, Technique
from sculptor.kg.store import SculptorKG, default_db_path

app = typer.Typer(
    name="sculpt",
    help=(
        "Reward Sculptor — autonomous iteration over RL reward functions, "
        "grounded in a knowledge graph of robotics papers."
    ),
    no_args_is_help=True,
    add_completion=False,
)


# ── KG sub-app ─────────────────────────────────────────────────────────────
kg_app = typer.Typer(
    name="kg",
    help="Inspect the knowledge-graph store (papers, techniques, stats).",
    no_args_is_help=True,
)
app.add_typer(kg_app, name="kg")


world_app = typer.Typer(
    name="world",
    help="Author, inspect, and validate prompt-driven robot environments.",
    no_args_is_help=True,
)
app.add_typer(world_app, name="world")


@world_app.command("author")
def world_author(
    prompt: str = typer.Argument(
        ..., help="Natural-language environment and task description."),
    project_dir: Path = typer.Option(
        ..., "--project", "-p", help="Initialized Sculptor project."),
    robot: Optional[str] = typer.Option(
        None, "--robot", help="Robot capability ID; auto-select when omitted."),
    robot_descriptor: list[Path] = typer.Option(
        [], "--robot-descriptor",
        help="External RobotCapability JSON (repeatable)."),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Select every disclosed system default without prompting."),
    interactive: Optional[bool] = typer.Option(
        None, "--interactive/--no-interactive",
        help=("Force clarification prompts or select disclosed defaults "
              "headlessly. By default, terminal input is detected.")),
    timeout_defaults: bool = typer.Option(
        False, "--timeout-defaults",
        help="Select defaults and record timeout_default provenance."),
    kg_grounding: bool = typer.Option(
        True, "--kg-grounding/--no-kg-grounding",
        help=("Ground authoring in the shared knowledge graph "
              "(best-effort retrieval; never blocks authoring).")),
    json_out: bool = typer.Option(
        False, "--json", help="Emit the promoted result as JSON."),
):
    """Author, clarify, gate, materialize, and atomically promote a world."""
    import json as _json

    from sculptor.world.author import (
        CLARIFICATION_VERSION,
        ClarificationAnswer,
        ClarificationSubmission,
        apply_clarifications,
        author_environment,
        default_clarification_submission,
    )
    from sculptor.world.grounding import (
        gather_grounding,
        grounding_context,
        grounding_ids,
    )
    from sculptor.world.project import WorldProjectService

    try:
        grounding_items = gather_grounding(prompt) if kg_grounding else ()
        draft = author_environment(
            prompt, robot_capability_id=robot,
            robot_descriptor_paths=robot_descriptor,
            grounding=grounding_ids(grounding_items),
            grounding_context=grounding_context(grounding_items))
        import sys

        should_prompt = (
            interactive if interactive is not None else sys.stdin.isatty()
        )
        if yes or timeout_defaults or not should_prompt:
            if (not yes and not timeout_defaults and not should_prompt):
                typer.echo(
                    "[sculpt world] non-interactive input; selecting every "
                    "disclosed system default",
                    err=True,
                )
            submission = default_clarification_submission(
                draft, timeout=timeout_defaults)
        else:
            answers: list[ClarificationAnswer] = []
            total_pages = len(draft.clarification_plan.pages)
            for page in draft.clarification_plan.pages:
                typer.echo(
                    f"Clarification {page.page}/{total_pages} "
                    f"({len(page.questions)} questions)", err=json_out)
                for question in page.questions:
                    typer.echo(f"\n{question.prompt}", err=json_out)
                    choices = list(question.choices)
                    for index, choice in enumerate(choices, 1):
                        typer.echo(
                            f"  {index}. {choice.label}", err=json_out)
                    default_index = len(choices) + 1
                    typer.echo(
                        f"  {default_index}. {question.system_default_label}",
                        err=json_out,
                    )
                    while True:
                        selected = typer.prompt(
                            "Select", type=int, default=default_index,
                            err=json_out)
                        if 1 <= selected <= default_index:
                            break
                        typer.echo(
                            f"Select a number from 1 to {default_index}.",
                            err=True,
                        )
                    if selected == default_index:
                        answers.append(ClarificationAnswer(
                            question.question_id, "system_default",
                            source="default"))
                    else:
                        answers.append(ClarificationAnswer(
                            question.question_id,
                            choices[selected - 1].choice_id,
                            source="user"))
            submission = ClarificationSubmission(
                version=CLARIFICATION_VERSION,
                draft_hash=draft.draft_hash,
                question_set_hash=(
                    draft.clarification_plan.question_set_hash),
                answers=tuple(answers),
            )
        applied = apply_clarifications(draft, submission)
        from sculptor.world.project import evaluation_lineage_for

        lineage = evaluation_lineage_for(
            applied.world_spec, applied.task_spec)
        admitted = WorldProjectService(project_dir).admit_and_promote(
            world=applied.world_spec, task=applied.task_spec,
            clarifications=applied.clarification_ledger,
            evaluation_lineage=lineage,
            rejected_session_id=f"draft-{draft.draft_hash[:24]}",
        )
    except Exception as exc:
        typer.echo(f"world author failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    result = {
        "ok": True,
        "selection": admitted.promoted.selection.to_dict(),
        "capability_id": draft.capability_id,
        "draft_hash": draft.draft_hash,
        "result_hash": applied.result_hash,
        "evaluation_lineage": lineage,
        "admission": admitted.admission,
        "asset_dir": str(admitted.asset_dir),
        "clarification_answers": len(
            applied.clarification_ledger.get("answers", [])),
        "kg_grounding": grounding_ids(grounding_items),
    }
    if json_out:
        typer.echo(_json.dumps(result, indent=2, sort_keys=True))
    else:
        selection_result = admitted.promoted.selection
        typer.echo(
            f"[sculpt world] promoted selection_v"
            f"{selection_result.selection_version} "
            f"({selection_result.tuple_hash[:12]})")
        typer.echo(f"  robot:       {draft.capability_id}")
        typer.echo(f"  lineage:     {lineage}")
        typer.echo(f"  eval assets: {admitted.asset_dir}")
        typer.echo(
            f"  gates:       {len(admitted.admission['gates'])} passed")
        typer.echo(
            f"  grounding:   {len(grounding_items)} KG nodes")


def _world_selection_path(project_dir: Path, selection: Optional[Path]) -> Path:
    if selection is not None:
        return selection.expanduser().resolve()
    return (project_dir.expanduser().resolve() /
            "env" / "selection_current.json")


@world_app.command("show")
def world_show(
    project_dir: Path = typer.Option(
        ..., "--project", "-p", help="Sculptor project."),
    selection: Optional[Path] = typer.Option(
        None, "--selection", help="Pinned selection_vN.json; default current."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit the complete selected bundle."),
):
    """Show the exact authoritative world/task/evaluation tuple."""
    import json as _json

    from sculptor.world.project import load_selected_world

    path = _world_selection_path(project_dir, selection)
    try:
        _store, selected, bundle = load_selected_world(path)
    except Exception as exc:
        typer.echo(f"world show failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    if json_out:
        typer.echo(_json.dumps(bundle, indent=2, sort_keys=True))
        return
    world = bundle["world"]
    task = bundle["task"]
    shared = world["shared"]
    goal = task["shared"]["goal"]
    typer.echo(
        f"selection_v{selected.selection_version} "
        f"{selected.tuple_hash[:12]}  lineage={selected.evaluation_lineage}")
    typer.echo(
        f"robot:   {shared['robot']['capability_id']} "
        f"requires={shared['robot'].get('required_capabilities', [])}")
    typer.echo(
        f"terrain: {shared['terrain']['kind']}  "
        f"objects={len(shared.get('objects', {}))}  "
        f"course={len(shared.get('obstacles', {}).get('course', []))}")
    typer.echo(
        f"goal:    {goal['type']} ({goal['id']})  "
        f"admission={bundle['resolved_eval']['admission']['ok']}")


@world_app.command("validate")
def world_validate(
    project_dir: Path = typer.Option(
        ..., "--project", "-p", help="Sculptor project."),
    selection: Optional[Path] = typer.Option(
        None, "--selection", help="Pinned selection_vN.json; default current."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit the complete gate report."),
):
    """Verify the immutable tuple, frozen assets, and runtime fingerprints."""
    import json as _json

    from sculptor.world.compiler import (
        ResolvedEvaluation,
        verify_resolved_evaluation,
    )
    from sculptor.world.gates import AdmissionReport
    from sculptor.world.project import load_selected_world

    path = _world_selection_path(project_dir, selection)
    try:
        store, selected, bundle = load_selected_world(path)
        manifest = ResolvedEvaluation.from_dict(bundle["resolved_eval"])
        verify_resolved_evaluation(
            bundle["world"], bundle["task"], bundle["channel_catalog"],
            manifest,
            asset_base=store.resolve_ref(
                selected.refs["resolved_eval"]).parent,
        )
        report = AdmissionReport.from_dict(manifest.admission)
        model_match = True
        ok = report.ok
    except Exception as exc:
        typer.echo(f"world validate failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    result = {
        "ok": ok,
        "selection_version": selected.selection_version,
        "tuple_hash": selected.tuple_hash,
        "evaluation_lineage": selected.evaluation_lineage,
        "model_hash_match": model_match,
        "admission": report.to_dict(),
    }
    if json_out:
        typer.echo(_json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"world validation {'passed' if ok else 'FAILED'}: "
            f"selection_v{selected.selection_version}")
        typer.echo(f"  tuple hash: {selected.tuple_hash}")
        typer.echo(f"  compiled model hash match: {model_match}")
        for gate in report.gates:
            typer.echo(
                f"  [{'ok' if gate.ok else 'FAIL'}] {gate.gate}"
                + (f" ({len(gate.violations)} violations)"
                   if gate.violations else ""))
    if not ok:
        raise typer.Exit(1)


# ── Remote sub-app (§Ship 23) ────────────────────────────────────────────────
remote_app = typer.Typer(
    name="remote",
    help="Remote GPU dispatch — connectivity / environment checks.",
    no_args_is_help=True,
)
app.add_typer(remote_app, name="remote")


@remote_app.command("doctor")
def remote_doctor(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help=(
            "config.toml with a [remote] table. SCULPTOR_REMOTE_* env vars "
            "override it; with no --config, env vars alone are used."
        ),
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Machine-readable output (used by the UI backend)."
    ),
):
    """Check the configured remote GPU host: ssh reachability, rsync,
    python, NVIDIA driver (>= R570 for Blackwell), torch.cuda,
    torch/mjlab version skew vs local, free disk. Exit 0 = all green,
    1 = at least one check failed, 2 = no remote configured."""
    import dataclasses as _dc
    import json as _json
    import os

    from sculptor.adapters._remote import RemoteConfig, RemoteExecutor

    table = None
    if config is not None:
        import tomllib

        try:
            with Path(config).open("rb") as f:
                table = tomllib.load(f).get("remote")
        except FileNotFoundError:
            typer.echo(f"config not found: {config}", err=True)
            raise typer.Exit(2)
        except tomllib.TOMLDecodeError as e:
            typer.echo(f"config not parseable: {config}: {e}", err=True)
            raise typer.Exit(2)
        except OSError as e:
            typer.echo(f"config not readable: {config}: {e}", err=True)
            raise typer.Exit(2)
    cfg = RemoteConfig.from_sources(table, os.environ)
    if cfg is None or not cfg.host:
        typer.echo(
            "no remote configured — add a [remote] table (host / user / "
            "key_path) to config.toml or set SCULPTOR_REMOTE_HOST.",
            err=True,
        )
        raise typer.Exit(2)
    if not cfg.enabled:
        # Doctor must work BEFORE the user flips enabled=true.
        cfg = _dc.replace(cfg, enabled=True)
    report = RemoteExecutor(cfg).doctor()
    if json_out:
        typer.echo(_json.dumps(report, indent=2))
    else:
        typer.echo(f"remote doctor — {cfg.target}:{cfg.port}")
        for c in report["checks"]:
            mark = " ok " if c["ok"] else "FAIL"
            typer.echo(f"  [{mark}] {c['name']}: {c['detail']}")
        typer.echo(
            "all checks passed" if report["ok"] else "some checks FAILED"
        )
    raise typer.Exit(0 if report["ok"] else 1)


# ── Eval sub-app (§Ship 27 / E2) ────────────────────────────────────────────
eval_app = typer.Typer(
    name="eval",
    help="Research-grade evaluation: seeds × conditions × benchmarks.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(
    out: Path = typer.Option(..., "--out", help="Campaign output dir."),
    benchmark: list[str] = typer.Option(
        ..., "--benchmark", "-b",
        help="Benchmark name (repeatable). See `sculpt eval list`.",
    ),
    benchmark_manifest: list[Path] = typer.Option(
        [], "--benchmark-manifest",
        help="Strict external benchmark-suite JSON (repeatable).",
    ),
    condition: list[str] = typer.Option(
        ..., "--condition", "-c",
        help="Condition name (repeatable): full | no_kg | plain_ppo | seed_only.",
    ),
    seeds: int = typer.Option(
        3, "--seeds", help="Number of paired seeds (1000, 1017, 1034, ...).",
    ),
    iterations: int = typer.Option(
        2, "--iterations", help="LLM-loop iterations for sculpt-mode conditions.",
    ),
    steps_per_iter: int = typer.Option(
        300, "--steps-per-iter", help="rsl_rl iterations per training run.",
    ),
    rollout_episodes: int = typer.Option(4, "--rollout-episodes"),
    spec_threshold: float = typer.Option(
        0.5, "--spec-threshold",
        help="spec_score threshold for iterations-to-criterion.",
    ),
    eureka_k: int = typer.Option(
        4, "--eureka-k",
        help="Eureka condition: candidates per generation "
             "(generations = --iterations).",
    ),
    fitness_in_loop: bool = typer.Option(
        False, "--fitness-in-loop",
        help="Feed the benchmark spec metric INTO the sculpt loop as a "
             "ground-truth fitness signal (fitness-guided diagnose + "
             "best-by-fitness selection + plateau early-stop). Removes the "
             "eval asymmetry where only eureka could select on fitness.",
    ),
    name: Optional[str] = typer.Option(
        None, "--name", help="Campaign name (default: out dir name)."),
    require_remote: bool = typer.Option(
        False, "--require-remote",
        help="Abort unless SCULPTOR_REMOTE_* resolves to an enabled "
             "remote host — guards a campaign against silently training "
             "on the local GPU when the env didn't propagate.",
    ),
):
    """Run an eval campaign. Resumable: completed jobs (result.json on
    disk) are skipped, so re-running after a crash or pod restart only
    does the remaining work. Remote dispatch: export SCULPTOR_REMOTE_*
    (see docs/remote.md) and training routes through the pod
    automatically."""
    import os as _os

    from sculptor.adapters._remote import RemoteConfig
    from sculptor.eval import CampaignConfig, run_campaign

    rcfg = RemoteConfig.from_sources(None, _os.environ)
    if rcfg is not None and rcfg.enabled:
        typer.echo(
            f"[eval] training target: REMOTE {rcfg.target}:{rcfg.port} "
            f"(device={rcfg.device or 'cuda:0'})"
        )
    else:
        typer.echo("[eval] training target: LOCAL GPU")
        if require_remote:
            typer.echo(
                "[eval] --require-remote set but no enabled remote "
                "resolved from SCULPTOR_REMOTE_* — aborting before any "
                "GPU work.", err=True,
            )
            raise typer.Exit(3)

    cfg = CampaignConfig(
        name=name or Path(out).name,
        out_dir=Path(out),
        benchmarks=list(benchmark),
        conditions=list(condition),
        seeds=[1000 + 17 * i for i in range(int(seeds))],
        benchmark_manifests=list(benchmark_manifest),
        iterations=iterations,
        steps_per_iter=steps_per_iter,
        rollout_episodes=rollout_episodes,
        spec_threshold=spec_threshold,
        eureka_k=eureka_k,
        fitness_in_loop=fitness_in_loop,
    )
    report = run_campaign(cfg)
    typer.echo(f"report: {Path(out) / 'campaign_report.json'}")
    typer.echo(f"html:   {Path(out) / 'report.html'}")
    for w in report["aggregates"]["capture_parity_warnings"]:
        typer.echo(f"WARNING: {w}", err=True)
    for w in report.get("authority_warnings", []):
        typer.echo(f"WARNING: {w}", err=True)


shard_app = typer.Typer(
    name="shard",
    help="Run one frozen eval matrix safely across processes or pods.",
    no_args_is_help=True,
)
eval_app.add_typer(shard_app, name="shard")


@shard_app.command("prepare")
def eval_shard_prepare(
    out: Path = typer.Option(..., "--out", help="Global campaign output dir."),
    shards: int = typer.Option(
        ..., "--shards", min=1, help="Number of operational worker shards.",
    ),
    benchmark: list[str] = typer.Option(
        ..., "--benchmark", "-b", help="Benchmark name (repeatable).",
    ),
    benchmark_manifest: list[Path] = typer.Option(
        [], "--benchmark-manifest",
        help="Strict external benchmark-suite JSON (repeatable).",
    ),
    condition: list[str] = typer.Option(
        ..., "--condition", "-c", help="Condition name (repeatable).",
    ),
    seeds: int = typer.Option(
        3, "--seeds", min=1,
        help="Number of paired seeds (1000, 1017, 1034, ...).",
    ),
    iterations: int = typer.Option(2, "--iterations", min=1),
    steps_per_iter: int = typer.Option(300, "--steps-per-iter", min=1),
    rollout_episodes: int = typer.Option(4, "--rollout-episodes", min=1),
    spec_threshold: float = typer.Option(0.5, "--spec-threshold"),
    eureka_k: int = typer.Option(4, "--eureka-k", min=1),
    fitness_in_loop: bool = typer.Option(False, "--fitness-in-loop"),
    name: Optional[str] = typer.Option(None, "--name"),
):
    """Charter the full matrix once and emit self-contained shard dirs."""
    from sculptor.eval import CampaignConfig
    from sculptor.eval.charter import CharterError
    from sculptor.eval.sharding import ShardError, prepare_sharded_campaign

    cfg = CampaignConfig(
        name=name or Path(out).name,
        out_dir=Path(out),
        benchmarks=list(benchmark),
        conditions=list(condition),
        seeds=[1000 + 17 * index for index in range(int(seeds))],
        benchmark_manifests=list(benchmark_manifest),
        iterations=iterations,
        steps_per_iter=steps_per_iter,
        rollout_episodes=rollout_episodes,
        spec_threshold=spec_threshold,
        eureka_k=eureka_k,
        fitness_in_loop=fitness_in_loop,
    )
    try:
        coordinator = prepare_sharded_campaign(cfg, shard_count=shards)
    except (CharterError, ShardError, OSError, ValueError, KeyError) as exc:
        typer.echo(f"shard preparation failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"global charter: {Path(out) / 'campaign_charter.json'}")
    typer.echo(f"coordinator:    {Path(out) / 'campaign_shards.json'}")
    typer.echo(f"design hash:    {coordinator['charter']['design_sha256']}")
    for record in coordinator["shards"]:
        manifest = (
            Path(out) / record["relative_dir"] / "shard_manifest.json"
        )
        typer.echo(
            f"{record['shard_id']}: {record['n_jobs']} jobs -> {manifest}"
        )


@shard_app.command("run")
def eval_shard_run(
    manifest: Path = typer.Argument(..., help="Transported shard_manifest.json."),
    require_remote: bool = typer.Option(
        False, "--require-remote",
        help="Abort unless an enabled SCULPTOR_REMOTE_* target resolves.",
    ),
):
    """Run only a shard's assigned jobs under its complete global charter."""
    import os as _os

    from sculptor.adapters._remote import RemoteConfig
    from sculptor.eval.charter import CharterError
    from sculptor.eval.sharding import ShardError, run_campaign_shard

    remote = RemoteConfig.from_sources(None, _os.environ)
    if require_remote and (remote is None or not remote.enabled):
        typer.echo(
            "[eval shard] --require-remote set but no enabled remote resolved; "
            "aborting before training.",
            err=True,
        )
        raise typer.Exit(3)
    try:
        report = run_campaign_shard(manifest)
    except (CharterError, ShardError, OSError, ValueError, KeyError) as exc:
        typer.echo(f"shard run failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    coverage = report["coverage"]
    typer.echo(
        f"{report['shard_id']}: {coverage['n_completed']}/"
        f"{coverage['n_expected']} assigned jobs complete"
    )
    typer.echo(f"report: {Path(manifest).parent / 'shard_report.json'}")


@shard_app.command("merge")
def eval_shard_merge(
    out: Path = typer.Option(..., "--out", help="Global campaign output dir."),
    shard_dir: list[Path] = typer.Option(
        [], "--shard-dir",
        help="Fetched shard output dir (repeatable; default: local plan dirs).",
    ),
):
    """Verify available shards and merge results with explicit coverage."""
    from sculptor.eval.charter import CharterError
    from sculptor.eval.sharding import ShardError, merge_sharded_campaign

    try:
        report = merge_sharded_campaign(
            out, list(shard_dir) if shard_dir else None,
        )
    except (CharterError, ShardError, OSError, ValueError, KeyError) as exc:
        typer.echo(f"shard merge failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    coverage = report["coverage"]
    state = "COMPLETE" if coverage["complete"] else "INCOMPLETE"
    typer.echo(
        f"coverage {state}: {coverage['n_completed']}/"
        f"{coverage['n_expected']} jobs; {coverage['n_missing']} missing"
    )
    typer.echo(f"report: {Path(out) / 'campaign_report.json'}")
    for warning in report.get("authority_warnings", []):
        typer.echo(f"WARNING: {warning}", err=True)


@eval_app.command("report")
def eval_report(
    out: Path = typer.Argument(..., help="Campaign dir with job results."),
):
    """Re-aggregate an existing campaign dir (e.g. after hand-pruning a
    job or to regenerate the HTML) without running anything."""
    import json as _json

    from sculptor.eval import CampaignConfig, run_campaign  # noqa: F401
    from sculptor.eval.charter import (
        CHARTER_FILENAME,
        CharterError,
        load_and_verify_charter,
        verify_result_lineage,
    )
    from sculptor.eval.harness import _report_html, aggregate
    from sculptor.run_context import write_json_atomic

    results = []
    for rp in sorted(Path(out).glob("*/*/seed_*/result.json")):
        results.append(_json.loads(rp.read_text(encoding="utf-8")))
    if not results:
        typer.echo(f"no result.json files under {out}", err=True)
        raise typer.Exit(1)
    charter_path = Path(out) / CHARTER_FILENAME
    try:
        charter = load_and_verify_charter(charter_path)
        verify_result_lineage(results, charter)
    except CharterError as exc:
        typer.echo(f"campaign integrity check failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    report_path = Path(out) / "campaign_report.json"
    try:
        prior = _json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        prior = {"name": Path(out).name, "config": {}}
    from datetime import datetime, timezone

    prior["created_at"] = datetime.now(timezone.utc).isoformat()
    prior["aggregates"] = aggregate(results)
    prior["jobs"] = [
        {k: v for k, v in r.items() if k != "spec_series"} for r in results
    ]
    write_json_atomic(report_path, prior)
    (Path(out) / "report.html").write_text(
        _report_html(prior), encoding="utf-8",
    )
    typer.echo(f"re-aggregated {len(results)} jobs -> {report_path}")


@eval_app.command("charter")
def eval_charter(
    out: Path = typer.Argument(..., help="Campaign directory to verify."),
):
    """Verify and summarize a frozen campaign charter and result lineage."""
    import json as _json

    from sculptor.eval.charter import (
        CHARTER_FILENAME,
        CharterError,
        load_and_verify_charter,
        verify_result_lineage,
    )

    charter_path = Path(out) / CHARTER_FILENAME
    try:
        charter = load_and_verify_charter(charter_path)
        results = [
            _json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(Path(out).glob("*/*/seed_*/result.json"))
        ]
        verify_result_lineage(results, charter)
    except (CharterError, OSError, _json.JSONDecodeError) as exc:
        typer.echo(f"campaign integrity check failed: {exc}", err=True)
        raise typer.Exit(2) from exc

    campaign = charter["design"]["campaign"]
    typer.echo("campaign charter: VERIFIED")
    typer.echo(f"  path:        {charter_path}")
    typer.echo(f"  campaign:    {campaign['name']}")
    typer.echo(f"  created:     {charter['created_at']}")
    typer.echo(f"  design hash: {charter['design_sha256']}")
    typer.echo(f"  source hash: "
               f"{charter['design']['runtime_identity']['source_tree_sha256']}")
    typer.echo(f"  results:     {len(results)} verified")


@eval_app.command("spec-audit")
def eval_spec_audit(
    manifest: Path = typer.Argument(
        ..., help="Frozen adversarial spec-audit manifest JSON.",
    ),
    out: Path = typer.Option(..., "--out", help="Fresh certificate output dir."),
):
    """Run an adversarial evidence battery for one objective spec metric."""
    import json as _json

    from sculptor.eval.spec_audit import SpecAuditError, run_spec_audit

    try:
        certificate = run_spec_audit(manifest, out)
    except SpecAuditError as exc:
        typer.echo(f"spec audit failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(_json.dumps({
        "audit_id": certificate["audit_id"],
        "spec_name": certificate["spec_name"],
        "passed": certificate["passed"],
        "authority_decision": certificate["authority_decision"],
        "coverage": certificate["coverage"],
        "summary": certificate["summary"],
        "certificate_sha256": certificate["certificate_sha256"],
        "report": str(Path(out) / "spec_audit_report.md"),
    }, indent=2))
    if not certificate["passed"]:
        raise typer.Exit(1)


gauntlet_app = typer.Typer(
    name="gauntlet",
    help="Build and analyze blinded evaluator/human-anchor studies.",
    no_args_is_help=True,
)
eval_app.add_typer(gauntlet_app, name="gauntlet")


@gauntlet_app.command("build")
def eval_gauntlet_build(
    manifest: Path = typer.Argument(
        ..., help="Private labeled source-manifest JSON.",
    ),
    out: Path = typer.Option(..., "--out", help="Fresh study output directory."),
    seed: int = typer.Option(0, "--seed", help="Frozen pairing/randomization seed."),
    forms: int = typer.Option(
        2, "--forms", min=1, max=2,
        help="One randomized form or two counterbalanced forms.",
    ),
    max_pairs_per_group: int = typer.Option(
        50, "--max-pairs-per-group", min=1,
        help="Balanced cap within each comparison group.",
    ),
    reliability_repeats: int = typer.Option(
        0, "--reliability-repeats", min=0,
        help="Hidden repeated pairs used to estimate rater self-consistency.",
    ),
    evaluator_tie_band: float = typer.Option(
        0.0, "--evaluator-tie-band", min=0.0,
        help="Absolute score difference treated as an evaluator tie.",
    ),
):
    """Build anonymized media packets plus a separate private study key."""
    import json as _json

    from sculptor.eval.gauntlet import GauntletError, build_blind_study

    try:
        summary = build_blind_study(
            manifest, out,
            seed=seed,
            forms=forms,
            max_pairs_per_group=max_pairs_per_group,
            reliability_repeats=reliability_repeats,
            evaluator_tie_band=evaluator_tie_band,
        )
    except GauntletError as exc:
        typer.echo(f"gauntlet build failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(_json.dumps(summary, indent=2))
    typer.echo(
        "Keep study_key.json private until labels are frozen; distribute only "
        "one study_packet_form_*.json and its referenced assets per rater."
    )


@gauntlet_app.command("analyze")
def eval_gauntlet_analyze(
    study_key: Path = typer.Argument(..., help="Private study_key.json."),
    responses: Path = typer.Argument(..., help="Completed JSONL responses."),
    out: Path = typer.Option(..., "--out", help="Analysis output directory."),
):
    """Validate frozen responses and analyze evaluator-human agreement."""
    import json as _json

    from sculptor.eval.gauntlet import GauntletError, analyze_blind_study

    try:
        analysis = analyze_blind_study(study_key, responses, out)
    except (GauntletError, OSError) as exc:
        typer.echo(f"gauntlet analysis failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(_json.dumps({
        "study_id": analysis["study_id"],
        "analysis_sha256": analysis["analysis_sha256"],
        "counts": analysis["counts"],
        "human_pair_accuracy": analysis["human"][
            "pair_majority_accuracy_vs_expected"
        ],
        "evaluator_human_alignment": analysis["evaluator"][
            "pair_accuracy_vs_human_majority"
        ],
        "report": str(Path(out) / "gauntlet_analysis.md"),
    }, indent=2))


@eval_app.command("list")
def eval_list(
    benchmark_manifest: list[Path] = typer.Option(
        [], "--benchmark-manifest",
        help="Include a strict external benchmark-suite JSON (repeatable).",
    ),
):
    """List benchmarks + conditions."""
    from sculptor.eval import CONDITIONS
    from sculptor.eval.benchmarks import (
        BenchmarkManifestError,
        benchmark_registry,
    )

    try:
        benchmarks = benchmark_registry(benchmark_manifest)
    except BenchmarkManifestError as exc:
        typer.echo(f"benchmark manifest invalid: {exc}", err=True)
        raise typer.Exit(2) from exc

    typer.echo("benchmarks:")
    for b in benchmarks.values():
        readiness = "ready" if b.campaign_ready else b.evaluation_tier
        typer.echo(
            f"  {b.name:22s} {b.embodiment_family:18s} "
            f"{readiness:16s} authority={b.spec_authority:18s} "
            f"spec={b.spec_metric or '-'} task={b.task_id}"
        )
    typer.echo("conditions:")
    for c in CONDITIONS.values():
        typer.echo(f"  {c.name:12s} mode={c.mode:11s} {c.notes}")


_STORE_OPT = typer.Option(
    None,
    "--store",
    help="Path to the KG DB (default: $SCULPTOR_KG_PATH / $RS_KG_PATH, else "
         "the shared ~/.local/share/sculptor/kg/graph.db).",
)


def _open_store(store: Optional[Path]) -> SculptorKG:
    return SculptorKG(store) if store else SculptorKG()


@kg_app.command("list-papers")
def kg_list_papers(store: Optional[Path] = _STORE_OPT):
    """List every Paper node in the store."""
    with _open_store(store) as kg:
        papers: list[Paper] = kg.find_nodes(kind=Paper.kind)  # type: ignore[assignment]
        if not papers:
            typer.echo(f"(no papers in {kg.db_path})")
            return
        papers.sort(key=lambda p: (p.year or 0, p.arxiv_id))
        typer.echo(f"{len(papers)} paper(s) in {kg.db_path}:\n")
        typer.echo(f"  {'arxiv_id':<13} {'year':<4} {'extracted':<9} title")
        typer.echo(f"  {'-'*13} {'-'*4} {'-'*9} {'-'*60}")
        for p in papers:
            title = (p.title[:70] + "…") if len(p.title) > 71 else p.title
            ext = "yes" if p.extracted else "no"
            typer.echo(f"  {p.arxiv_id:<13} {str(p.year or '?'):<4} {ext:<9} {title}")


@kg_app.command("list-techniques")
def kg_list_techniques(store: Optional[Path] = _STORE_OPT):
    """List every Technique node in the store."""
    with _open_store(store) as kg:
        techniques: list[Technique] = kg.find_nodes(kind=Technique.kind)  # type: ignore[assignment]
        if not techniques:
            typer.echo(f"(no techniques in {kg.db_path})")
            return
        techniques.sort(key=lambda t: t.name.lower())
        typer.echo(f"{len(techniques)} technique(s) in {kg.db_path}:\n")
        for t in techniques:
            tags = (", ".join(t.tags)) if t.tags else "—"
            desc = t.description[:80] + ("…" if len(t.description) > 80 else "")
            typer.echo(f"  - {t.name}")
            typer.echo(f"      tags: {tags}")
            if desc:
                typer.echo(f"      {desc}")


@kg_app.command("stats")
def kg_stats(store: Optional[Path] = _STORE_OPT):
    """Print node/edge counts by type."""
    with _open_store(store) as kg:
        s = kg.stats()
        typer.echo(f"KG @ {s['db_path']}")
        typer.echo(f"  total_nodes:      {s['total_nodes']}")
        typer.echo(f"  total_edges:      {s['total_edges']}")
        typer.echo(f"  total_embeddings: {s.get('total_embeddings', 0)}")
        typer.echo("  nodes_by_kind:")
        if s["nodes_by_kind"]:
            for kind, n in sorted(s["nodes_by_kind"].items()):
                typer.echo(f"    {kind:<20} {n}")
        else:
            typer.echo("    (none)")
        typer.echo("  edges_by_relation:")
        if s["edges_by_relation"]:
            for rel, n in sorted(s["edges_by_relation"].items()):
                typer.echo(f"    {rel:<20} {n}")
        else:
            typer.echo("    (none)")


@kg_app.command("merge")
def kg_merge(
    source: Path = typer.Argument(
        ..., exists=True, readable=True,
        help="Path to a stray/legacy graph.db to merge INTO the shared KG."),
    store: Optional[Path] = _STORE_OPT,
    rename_source: bool = typer.Option(
        True, "--rename-source/--keep-source",
        help="After a successful merge, rename the source to "
             "<name>.merged so it can never re-fragment the graph."),
):
    """Merge a stray per-directory KG into the shared graph (additive:
    existing shared nodes are never overwritten; edges/embeddings dedupe).

    Context: pre-2026-07-03 the default DB resolution preferred a
    cwd-relative kg/graph.db, silently splitting papers/techniques/run
    cases by launch directory. This command folds those strays back in.
    """
    from sculptor.kg.store import merge_stores

    with _open_store(store) as kg:
        counts = merge_stores(source, kg)
    typer.echo(
        f"merged {source} -> {kg.db_path}: "
        f"+{counts['nodes']} nodes ({counts['nodes_skipped']} already "
        f"present), +{counts['edges']} edges, "
        f"+{counts['embeddings']} embeddings")
    if rename_source:
        target = source.with_suffix(source.suffix + ".merged")
        source.rename(target)
        typer.echo(f"source renamed to {target}")


@kg_app.command("heal-stubs")
def kg_heal_stubs(store: Optional[Path] = _STORE_OPT):
    """§7.7: re-ingest Paper nodes whose title is still `arxiv:XXXX.XXXXX`.

    Stubs get left behind when the arxiv API rate-limits a bulk ingest —
    the fallback path writes `arxiv:<id>` as a placeholder title. This
    command scans the KG, finds every such Paper, and re-calls
    `ingest_arxiv(id, force=True)` on each. Runs in ~2 min for ~10
    stubs on a healthy network; safe to re-run (papers whose title
    filled in are skipped on subsequent runs).
    """
    from sculptor.kg.ingest import heal_stub_titles

    with _open_store(store) as kg:
        results = heal_stub_titles(store=kg)
        if not results:
            typer.echo("no stub-titled papers found — KG is clean")
            return
        healed = sum(1 for v in results.values() if v == "healed")
        stubbed = sum(1 for v in results.values() if v == "still_stubbed")
        errored = len(results) - healed - stubbed
        typer.echo(
            f"heal: {healed} healed, {stubbed} still stubbed, {errored} errored"
        )
        if stubbed:
            typer.echo(
                "  tip: still-stubbed papers probably hit arxiv rate-limit; "
                "re-run after ~2 min"
            )


@kg_app.command("index-fulltext")
def kg_index_fulltext(store: Optional[Path] = _STORE_OPT):
    """Index every Paper's stored body for lexical retrieval.

    Paper search embeds `title + abstract + rationale` only, so a paper that
    answers a question in its body but never says so in its abstract could not
    be retrieved — and extraction only ever summarized the first ~28K chars of
    each paper into the graph. This builds the recall path over the rest.

    Local and fast (no network, no LLM); safe to re-run — re-indexing a paper
    replaces its row. Run it after any bulk ingest.
    """
    from sculptor.kg.ingest import backfill_full_text_index

    with _open_store(store) as kg:
        res = backfill_full_text_index(store=kg)
        typer.echo(
            f"full-text index: {res['indexed']} indexed, "
            f"{res['missing']} missing a body, {res['skipped']} skipped"
        )
        if res["skipped"]:
            typer.echo(
                "  note: skipped means this SQLite build lacks FTS5 — "
                "retrieval falls back to abstract-only ranking"
            )
        if res["missing"]:
            typer.echo(
                "  tip: papers missing a body were ingested without a PDF; "
                "`sculpt kg doctor` lists dead full_text_path entries"
            )


@kg_app.command("doctor")
def kg_doctor(
    store: Optional[Path] = _STORE_OPT,
    fix: bool = typer.Option(
        False, "--fix",
        help="Repair mechanical issues: delete orphan embeddings + dangling "
             "edges, re-embed missing/stale vectors, heal stub titles and "
             "dead full_text_path sidecars (the last two need network)."),
    reembed_all: bool = typer.Option(
        False, "--reembed-all",
        help="With --fix: drop and rebuild EVERY semantic-pool embedding "
             "(Paper/Technique/FailureMode/RunCase) — the escape hatch for "
             "pre-hash vectors whose staleness is unknowable."),
    no_network: bool = typer.Option(
        False, "--no-network",
        help="With --fix: skip the two repairs that hit arxiv."),
):
    """§Phase-0 hardening: full KG integrity report (referential slack,
    stub/dead papers, missing + stale embeddings), optionally repaired.

    Read-only without --fix. Exit code 1 when unfixed issues remain."""
    from sculptor.kg.doctor import format_report, run_doctor

    with _open_store(store) as kg:
        report = run_doctor(
            kg, fix=fix, reembed_all=reembed_all, network=not no_network)
        typer.echo(format_report(report))
        effective = report.get("post_fix", report)
        dirty = bool(
            effective["dangling_edges"] or effective["orphan_embeddings"]
            or effective["unknown_kind_nodes"]
            or effective["unknown_relation_edges"]
            or effective["stub_titled_papers"] or effective["dead_text_paths"]
            or any(c["missing"] or c["stale"]
                   for c in effective["embedding_pools"].values()))
        if dirty:
            raise typer.Exit(code=1)


@kg_app.command("viz")
def kg_viz(
    out: Path = typer.Option(
        Path("kg.html"), "--out", "-o",
        help="Output HTML path for the pyvis graph."),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Project config.toml. If given, reports/provenance.json is "
             "loaded and nodes cited in that file are highlighted."),
    store: Optional[Path] = _STORE_OPT,
    title: str = typer.Option(
        "Reward Sculptor — Knowledge Graph", "--title",
        help="Page title + legend heading."),
):
    """Render the KG to an interactive HTML via pyvis."""
    import json as _json

    from sculptor.kg.viz import build_kg_html

    provenance: Optional[dict] = None
    if config is not None:
        prov_path = config.resolve().parent / "reports" / "provenance.json"
        if prov_path.is_file():
            try:
                provenance = _json.loads(prov_path.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                typer.echo(
                    f"[kg viz] warning: could not load {prov_path}: {e}")
        else:
            typer.echo(
                f"[kg viz] {prov_path} not found — rendering without "
                "provenance highlighting.")

    with _open_store(store) as kg:
        result = build_kg_html(
            kg, out_path=out, provenance=provenance, title=title)
    typer.echo(
        f"[kg viz] wrote {result.out_path}  "
        f"(nodes={result.n_nodes}, edges={result.n_edges}, "
        f"active={result.n_active_nodes})")


@kg_app.command("extract")
def kg_extract(
    all_: bool = typer.Option(
        False, "--all", help="Extract every Paper node with extracted=False."),
    force: bool = typer.Option(
        False, "--force", help="Re-extract papers even if already marked extracted."),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Cap number of papers processed this run."),
    seeds: Optional[Path] = typer.Option(
        None, "--seeds", exists=True, readable=True,
        help="Restrict extraction to Paper IDs in this campaign seeds YAML."),
    tier: Optional[str] = typer.Option(
        None, "--tier", help="Restrict to a structured tier (for example S)."),
    tag: Optional[str] = typer.Option(
        None, "--tag", help="Restrict to one structured campaign tag."),
    print_one: bool = typer.Option(
        True, "--print-one/--no-print-one",
        help="Dump the first successful payload as JSON for inspection."),
    store: Optional[Path] = _STORE_OPT,
):
    """Run LLM extraction over Papers in the KG.

    Requires `ANTHROPIC_API_KEY` in the environment. Creates Technique,
    FailureMode, RewardComponent, Environment nodes and their edges.
    """
    from sculptor.kg.extract import cli_extract_all, paper_ids_from_seeds

    if not (all_ or force or seeds or tier or tag):
        typer.echo("specify --all, --seeds, --tier, or --tag")
        raise typer.Exit(code=2)
    paper_ids = (
        paper_ids_from_seeds(seeds, tier=tier, tag=tag) if seeds else None)
    if seeds and not paper_ids:
        typer.echo("selection matched no papers")
        raise typer.Exit(code=2)
    raise typer.Exit(code=cli_extract_all(
        store, force=force, limit=limit, print_one=print_one,
        paper_ids=paper_ids, tier=(None if seeds else tier),
        tags=({tag} if tag and not seeds else None)))


@app.command()
def init(
    project_dir: Path = typer.Argument(
        ..., help="Directory to scaffold. Must be empty or non-existent."),
    adapter: str = typer.Option(
        "gym_sb3",
        "--adapter",
        help="Adapter short name (gym_sb3) or dotted class path "
             "(sculptor.adapters.gym_sb3.GymSB3Adapter).",
    ),
):
    """Scaffold a new Sculptor project: config.toml, rewards/v0.py,
    kg_seeds.yml, .gitignore, and an initial git commit."""
    from sculptor.sculpt import sculpt_init

    created = sculpt_init(project_dir, adapter=adapter)
    typer.echo(f"[sculpt init] scaffolded {created}")
    typer.echo(f"  config:  {created / 'config.toml'}")
    typer.echo(f"  reward:  {created / 'rewards' / 'v0.py'}")
    typer.echo(f"  seeds:   {created / 'kg_seeds.yml'}")
    typer.echo("Next: edit config.toml for your env + populate kg_seeds.yml, "
               "then `sculpt run`.")


def _warn_fitness_metric_mismatch(
    config_path: Path, fitness_metric: Optional[str],
) -> None:
    """§Ship 34: emit a visible, non-blocking warning if the chosen
    fitness metric looks mismatched to the project's robot/task (e.g.
    go1_trot on a G1 task). Surfaces in the UI log stream so the user can
    abort before wasting GPU on a semantically-wrong objective."""
    if not fitness_metric:
        return
    from sculptor.eval.spec_metrics import spec_metric_robot_warning

    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - py310
            import tomli as tomllib  # type: ignore[no-redef]
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        task_id = ((cfg.get("adapter") or {}).get("config") or {}).get("task_id")
    except Exception:  # noqa: BLE001 — the warning is best-effort
        task_id = None
    msg = spec_metric_robot_warning(fitness_metric, task_id)
    if msg:
        from sculptor.sculpt import _emit_event

        _emit_event({
            "type": "fitness_metric_warning",
            "message": msg,
            "fitness_metric": fitness_metric,
            "task_id": task_id,
        })
        typer.echo(f"[sculpt] WARNING: {msg}", err=True)


@app.command()
def run(
    behavior: str = typer.Argument(
        ..., help="Natural-language behavior goal, e.g., "
                  "'run forward as fast as possible without falling'."),
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, readable=True,
        help="Path to the project's config.toml."),
    iterations: int = typer.Option(
        10, "--iterations", "-n",
        help="Number of sculpt iterations to run this invocation."),
    resume_run: bool = typer.Option(
        False, "--resume",
        help="Start after the highest existing v<n>.py in rewards/."),
    no_kg: bool = typer.Option(
        False, "--no-kg",
        help="Skip KG queries; diagnoser sees an empty literature_context."),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Bypass all LLM calls; cap training at 1000 steps."),
    steps_per_iter: Optional[int] = typer.Option(
        None, "--steps-per-iter",
        help="Override [iteration].steps_per_iter from config.toml. "
             "For mjlab this is rsl_rl max_iterations; for gym_sb3 it's "
             "env steps per cycle. UI's 'rsl_rl iters / cycle' field "
             "maps to this."),
    num_envs: Optional[int] = typer.Option(
        None, "--num-envs", min=1, max=8192,
        help="Override [adapter].config.num_envs for this run. "
             "Supported by adapters that expose a parallel env count."),
    device: Optional[str] = typer.Option(
        None, "--device",
        help="Override the adapter device for this run (cpu, cuda, or "
             "cuda:N)."),
    # §Ship-7: rollout-video + RL knobs. Each defaults to None, meaning
    # the runner picks a sensible default (real-time video, 500-step
    # episodes, auto-rendered framerate). UI's Advanced tab surfaces
    # every flag below.
    max_episode_steps: Optional[int] = typer.Option(
        None, "--max-episode-steps",
        help="Rollout env steps per episode (default 500)."),
    playback_speed: Optional[float] = typer.Option(
        None, "--playback-speed",
        help="Video playback speed multiplier; 1.0 = real-time."),
    render_every: Optional[int] = typer.Option(
        None, "--render-every",
        help="Capture every N-th step (advanced; default auto-caps)."),
    rollout_fps: Optional[float] = typer.Option(
        None, "--rollout-fps",
        help="Hard override on playback fps (default: derive from step_dt)."),
    render_width: Optional[int] = typer.Option(
        None, "--render-width",
        help="Rollout video width in px (default 1280; render cost is "
             "resolution-independent on this stack)."),
    render_height: Optional[int] = typer.Option(
        None, "--render-height",
        help="Rollout video height in px (default 720)."),
    render_env_index: Optional[int] = typer.Option(
        None, "--render-env-index", min=0, max=63,
        help="Precommit which parallel evaluation lane is rendered in the "
             "video (0-63). Batch metrics remain unchanged."),
    rollout_episodes: Optional[int] = typer.Option(
        None, "--rollout-episodes",
        help="Override [iteration].rollout_episodes (default 6)."),
    seed: Optional[int] = typer.Option(
        None, "--seed",
        help="Base RNG seed (iter i uses seed + i). Overrides config."),
    auto_adjust_physics: Optional[bool] = typer.Option(
        None, "--auto-adjust-physics/--no-auto-adjust-physics",
        help="Enable/disable §7.4 physics-edit suggestion on severe realism verdicts."),
    early_stop_enabled: Optional[bool] = typer.Option(
        None, "--early-stop/--no-early-stop",
        help="Compatibility no-op: metric-plateau auto-kill is disabled."),
    early_stop_patience: Optional[int] = typer.Option(
        None, "--early-stop-patience",
        help="Compatibility no-op: accepted but ignored."),
    # §Ship 34: objective fitness-in-the-loop. `--fitness-metric` names a
    # spec metric (including capability-driven object_lift_hold); the
    # loop then best-selects on it, shows it to the diagnoser, and
    # plateau/target early-stops. None = blind (criterion/metric-history
    # only). The UI's "Objective fitness metric" dropdown maps here.
    fitness_metric: Optional[str] = typer.Option(
        None, "--fitness-metric",
        help="Spec-metric name to use as ground-truth fitness in the loop "
             "(go1_trot, g1_kick, g1_floss, cartpole_balance, "
             "object_lift_hold). Must match "
             "the robot. Omit for the blind loop."),
    fitness_target: Optional[float] = typer.Option(
        None, "--fitness-target",
        help="Stop once best fitness reaches this (0-1). Requires "
             "--fitness-metric."),
    fitness_patience: Optional[int] = typer.Option(
        None, "--fitness-patience",
        help="Stop after this many iters with no new best fitness "
             "(default 2). Requires --fitness-metric."),
    # §Ship 35: observe vs steer. observe = compute + display fitness but
    # DON'T let it influence the run (no diagnoser feed / best-selection /
    # early-stop) — for a fair blind-vs-guided A/B and for auto-generated
    # metrics that haven't earned steer-rights. steer = use it (default).
    fitness_mode: str = typer.Option(
        "steer", "--fitness-mode",
        help="'steer' (default): fitness drives selection/early-stop. "
             "'observe': compute + display only, no influence."),
    # §Ship 36: revert-on-regression. In steer mode, when an iter fails to
    # set a new best fitness the next iter rebuilds from the best-so-far
    # reward instead of the degraded latest (best-first search). Default on.
    fitness_revert: bool = typer.Option(
        True, "--fitness-revert/--no-fitness-revert",
        help="Steer mode: on a fitness regression, revert the edit base to "
             "the best-so-far reward instead of compounding the bad edit. "
             "Default on; --no-fitness-revert restores the Ship-33 behavior."),
    # §2026-07-04 (gap #7): warm-start chaining for complex motions —
    # e.g. learn a hop, then start the tuck-jump run from that policy.
    # Previously reachable only via the sculpt_run kwarg (the tuck-jump
    # E2E needed a hand-written driver script to use it).
    init_policy: Optional[Path] = typer.Option(
        None, "--init-policy", exists=True, readable=True,
        help="rsl_rl checkpoint to warm-start the FIRST iteration's "
             "training from (actor+critic only; optimizer/iteration "
             "state skipped). Task obs/action spaces must match."),
    # §Ship 39 (H1): interactive human-in-the-loop control.
    control_file: Optional[Path] = typer.Option(
        None, "--control-file",
        help="Interactive control sidecar (JSON: mode/resume_token/feedback/"
             "stop). When set, the loop pauses for human feedback at each "
             "iteration boundary while mode=='manual'. The UI writes it; omit "
             "for a fully-automated run."),
    feedback_timeout: float = typer.Option(
        3600.0, "--feedback-timeout",
        help="Max seconds to wait at an interactive pause before auto-resuming "
             "(so a dead client can't pin the GPU)."),
    reference_clip: Optional[str] = typer.Option(
        None, "--reference-clip",
        help="Reference-library clip id used as an immutable tracking prior."),
    reference_robot: Optional[str] = typer.Option(
        None, "--reference-robot",
        help="Exact reference-library robot namespace for --reference-clip."),
):
    """Run the inner loop: train → rollout → diagnose → edit → commit."""
    from sculptor.sculpt import sculpt_run

    if fitness_mode not in ("steer", "observe"):
        raise typer.BadParameter("--fitness-mode must be 'steer' or 'observe'")
    if device is not None and not re.fullmatch(r"(?:cpu|cuda(?::\d+)?)", device):
        raise typer.BadParameter(
            "--device must be 'cpu', 'cuda', or 'cuda:N'"
        )

    # Resolve the metric (built-in name or generated-metric path) to a
    # fitness fn (fail fast before any GPU work). None keeps the blind loop.
    fitness_fn = None
    if fitness_metric:
        from sculptor.eval import resolve_fitness_fn
        from sculptor.world.channels import load_project_channel_catalog

        channel_catalog = load_project_channel_catalog(config.parent)
        fitness_fn = resolve_fitness_fn(
            fitness_metric, channel_catalog=channel_catalog)
        _warn_fitness_metric_mismatch(config, fitness_metric)

    _sculpt_kwargs = dict(
        config_path=config, behavior_goal=behavior, iterations=iterations,
        resume=resume_run, no_kg=no_kg, dry_run=dry_run,
        steps_per_iter=steps_per_iter,
        num_envs=num_envs,
        device=device,
        max_episode_steps=max_episode_steps,
        playback_speed=playback_speed,
        render_every=render_every,
        rollout_fps=rollout_fps,
        render_width=render_width,
        render_height=render_height,
        render_env_index=render_env_index,
        rollout_episodes=rollout_episodes,
        seed=seed,
        auto_adjust_physics=auto_adjust_physics,
        early_stop_enabled=early_stop_enabled,
        early_stop_patience=early_stop_patience,
        fitness_fn=fitness_fn,
        fitness_observe_only=(fitness_mode == "observe"),
        fitness_revert=fitness_revert,
        init_policy_path=init_policy,
        control_file=control_file,
        feedback_timeout=feedback_timeout,
        reference_clip_id=reference_clip,
        reference_robot=reference_robot,
    )
    # Only override sculpt_run's defaults when explicitly provided.
    if fitness_target is not None:
        _sculpt_kwargs["fitness_target"] = fitness_target
    if fitness_patience is not None:
        _sculpt_kwargs["fitness_patience"] = fitness_patience
    result = sculpt_run(**_sculpt_kwargs)
    typer.echo(
        f"[sculpt] done. iters_run={result.iterations_run} "
        f"early_stopped={result.early_stopped}")
    if result.early_stopped:
        typer.echo(f"[sculpt] {result.early_stop_reason}")
    if result.final_reward_path:
        typer.echo(f"[sculpt] final reward: {result.final_reward_path}")


@app.command("gen-metric")
def gen_metric(
    goal: str = typer.Argument(
        ..., help="Behavior goal to generate an objective fitness metric for."),
    out: Path = typer.Option(
        ..., "--out", "-o",
        help="Directory to write metric.py + meta.json into."),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Optional project config.toml — its task_id is passed as a "
             "robot hint to the generator."),
    no_review: bool = typer.Option(
        False, "--no-review",
        help="Skip the independent-LLM review gate (validation still runs)."),
    calibrate_against: Optional[str] = typer.Option(
        None, "--calibrate-against",
        help="Built-in metric (including object_lift_hold) "
             "to calibrate the generated metric against (earns steer-rights "
             "if Spearman >= 0.7)."),
):
    """§Ship 35: auto-generate an OBJECTIVE fitness metric for a goal.

    Generate (LLM) -> validate (safety/contract/determinism/bounds/non-
    degeneracy) -> regenerate on failure -> independent review. The metric
    is OBSERVE-ONLY until calibrated; pass --calibrate-against to check it
    ranks like a hand-authored ground-truth metric."""
    from sculptor.eval import calibrate_metric, generate_objective_metric

    robot_hint: Optional[str] = None
    channel_catalog = None
    if config is not None:
        try:
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover - py310
                import tomli as tomllib  # type: ignore[no-redef]
            with open(config, "rb") as f:
                cfg = tomllib.load(f)
            robot_hint = ((cfg.get("adapter") or {}).get("config") or {}).get("task_id")
            from sculptor.world.channels import load_project_channel_catalog

            channel_catalog = load_project_channel_catalog(config.parent)
        except Exception:  # noqa: BLE001 — hint is best-effort
            robot_hint = None

    metric_kwargs = (
        {"channel_catalog": channel_catalog}
        if channel_catalog is not None else {}
    )
    result = generate_objective_metric(
        goal, out, robot_hint=robot_hint, review=not no_review,
        **metric_kwargs)
    typer.echo(f"[gen-metric] accepted={result['accepted']} "
               f"(validation_passed={result['validation_passed']})")
    typer.echo(f"[gen-metric] metric: {result['metric_path']}")
    if not result["accepted"]:
        reasons = (result.get("validation") or {}).get("reasons") or []
        for r in reasons:
            typer.echo(f"  - {r}", err=True)
        rev = result.get("review") or {}
        for c in rev.get("concerns", []):
            typer.echo(f"  - [review] {c}", err=True)
    if calibrate_against and result["accepted"]:
        cal = calibrate_metric(
            result["metric_path"], calibrate_against, **metric_kwargs)
        typer.echo(f"[gen-metric] calibration vs {calibrate_against}: "
                   f"spearman={cal.get('spearman')} ok={cal.get('ok')}")


@app.command()
def resume(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, readable=True),
    behavior: str = typer.Argument(
        ..., help="Behavior goal used for the resumed iterations."),
    iterations: int = typer.Option(10, "--iterations", "-n"),
):
    """Shortcut for `sculpt run --resume`."""
    from sculptor.sculpt import sculpt_run

    result = sculpt_run(
        config_path=config, behavior_goal=behavior, iterations=iterations,
        resume=True,
    )
    typer.echo(
        f"[sculpt] done. iters_run={result.iterations_run} "
        f"early_stopped={result.early_stopped}")


@app.command()
def viz(
    run_dir: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(
        Path("timelapse.html"),
        help="Output HTML (time-lapse + KG citations per iter).",
    ),
):
    """Render a time-lapse + changelog HTML for a completed (or live) run."""
    typer.echo(f"sculpt viz: not implemented yet (would write {out}).")


# ── Mission subcommands (Ship 18a) ───────────────────────────────────
@app.command("mission-init")
def mission_init(
    project_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Existing sculpt project directory (must contain config.toml).",
    ),
    goal: str = typer.Option(
        ..., "--goal", "-g",
        help="Behavior goal for the mission (Claude decomposes into stages).",
    ),
    mission_slug: Optional[str] = typer.Option(
        None, "--slug",
        help=(
            "Override the auto-derived slug. The mission lives at "
            "<project_dir>/.missions/<slug>/. Defaults to a slug "
            "derived from the goal."
        ),
    ),
    no_kg: bool = typer.Option(
        False, "--no-kg",
        help="Skip KG context to Claude (faster, less grounded).",
    ),
    no_skill_library: bool = typer.Option(
        False, "--no-skill-library",
        help=(
            "Skip the cross-mission skill library. Default: ON — Claude "
            "sees up to 5 prior-mission policies compatible with the "
            "project's adapter+task_id and may warm-start a stage from "
            "one. Ship 19."
        ),
    ),
    skill_library_root: Optional[Path] = typer.Option(
        None, "--skill-library-root",
        help=(
            "Override the on-disk root for the skill library. Default: "
            "$SCULPTOR_SKILL_LIBRARY_ROOT or ~/.local/share/sculptor/skills/."
        ),
    ),
    stage_metrics: bool = typer.Option(
        True, "--stage-metrics/--no-stage-metrics",
        help=(
            "§MISSION_METRIC_GRANULARITY: generate one trust-gated "
            "objective metric PER STAGE from the stage's goal text "
            "(default ON). Rejected generations leave the stage on the "
            "mission-level metric fallback."
        ),
    ),
):
    """Decompose a goal into a mission curriculum (Ship 14 + 17).

    Reads the project's config.toml to load the adapter, asks Claude
    to decompose the goal into 2-8 stages, validates the result, and
    writes `<project_dir>/.missions/<mission_slug>/mission.json`.
    Also emits `[SCULPT-EVENT] mission_initialized` to stdout so the
    backend's job-manager can capture mission_slug.
    """
    import json as _json
    from sculptor.adapters.base import load_adapter
    from sculptor.decompose import decompose_task
    from sculptor.kg.store import SculptorKG
    from sculptor.mission import save_mission

    config_path = project_dir / "config.toml"
    if not config_path.is_file():
        typer.echo(
            f"[mission-init] error: {config_path} not found — "
            "is this a sculpt project?", err=True,
        )
        raise typer.Exit(code=2)

    adapter = load_adapter(config_path)
    reward_contract = adapter.reward_contract()

    # Resolve mission slug: use explicit override OR derive + collide-resolve.
    missions_root = project_dir / ".missions"
    missions_root.mkdir(parents=True, exist_ok=True)
    existing_slugs = {p.name for p in missions_root.iterdir() if p.is_dir()}
    if mission_slug is None:
        mission_slug = _derive_mission_slug(goal, existing_slugs)
    elif mission_slug in existing_slugs:
        typer.echo(
            f"[mission-init] error: mission slug {mission_slug!r} "
            f"already exists in {missions_root}.",
            err=True,
        )
        raise typer.Exit(code=2)

    # §Ship 19: build a skill-library handle from the project's
    # adapter + task_id, unless `--no-skill-library` was passed.
    handle = (
        None if no_skill_library
        else _build_skill_library_handle(
            config_path, library_root=skill_library_root,
        )
    )

    # §llm provenance: archive the decompose call(s) to the mission dir
    # (created up front — decompose_task itself never writes into it).
    from sculptor.llm import set_llm_log_dir

    mission_dir = missions_root / mission_slug
    mission_dir.mkdir(parents=True, exist_ok=True)
    set_llm_log_dir(mission_dir)

    # Open KG (optional).
    kg_store = None if no_kg else SculptorKG()
    try:
        mission = decompose_task(
            goal, reward_contract, kg_store=kg_store,
            skill_library_handle=handle,
        )
    finally:
        if kg_store is not None:
            kg_store.close()

    mission.mission_dir = str(mission_dir.resolve())
    save_mission(mission, mission_dir)

    # §MISSION_METRIC_GRANULARITY: fresh trust-gated metric per stage,
    # generated from each stage's own goal text. After save_mission so a
    # generation crash can never lose the decomposition; re-saved after.
    if stage_metrics:
        from sculptor.mission_metrics import generate_stage_metrics

        robot_hint = getattr(adapter, "task_id", None)
        metric_kwargs = (
            {"channel_catalog": reward_contract.channel_catalog}
            if reward_contract.channel_catalog is not None else {}
        )
        report = generate_stage_metrics(
            mission, robot_hint=robot_hint, **metric_kwargs)
        save_mission(mission, mission_dir)
        typer.echo(
            f"[mission-init] stage metrics: "
            f"{len(report['generated'])} generated, "
            f"{len(report['rejected'])} rejected (fallback), "
            f"{len(report['skipped'])} skipped")
        for row in report["rejected"]:
            typer.echo(
                f"[mission-init]   {row['stage']}: {row['reason']}",
                err=True)
        print(
            "[SCULPT-EVENT] " + _json.dumps({
                "type": "mission_stage_metrics",
                "mission_slug": mission_slug,
                "generated": report["generated"],
                "rejected": report["rejected"],
            }),
            flush=True,
        )

    print(
        "[SCULPT-EVENT] " + _json.dumps({
            "type": "mission_initialized",
            "mission_slug": mission_slug,
            "n_stages": len(mission.stages),
            "mission_dir": str(mission_dir.resolve()),
        }),
        flush=True,
    )
    typer.echo(
        f"[mission-init] decomposed into {len(mission.stages)} stages "
        f"at {mission_dir}"
    )


def _build_skill_library_handle(
    config_path: Path,
    *,
    library_root: Optional[Path],
) -> Optional[Any]:
    """Construct a `SkillLibraryHandle` from a project's config.toml.

    Reads the adapter dotted-path + the `task_id` from `[adapter.config]`
    (mjlab convention; other adapters typically use `env_id` and won't
    match the library's lookup, but their writes are still gated on
    `adapter.train` accepting `init_policy_path` so nothing harmful
    is published).

    Returns None on read failure — the CLI continues without the
    library rather than crash, matching the "skill library is a
    soft feature" v1 stance.
    """
    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redef]
        with config_path.open("rb") as f:
            cfg = tomllib.load(f)
    except Exception as e:  # noqa: BLE001
        typer.echo(
            f"[skill-library] disabled — failed to read {config_path}: {e}",
            err=True,
        )
        return None

    adapter_section = (cfg.get("adapter") or {})
    adapter_class = (adapter_section.get("class") or "").strip()
    adapter_cfg = adapter_section.get("config") or {}
    task_id = (
        str(adapter_cfg.get("task_id")
            or adapter_cfg.get("env_id")
            or "").strip()
    )
    if not adapter_class or not task_id:
        typer.echo(
            "[skill-library] disabled — config.toml lacks "
            "[adapter].class or [adapter.config].task_id/env_id.",
            err=True,
        )
        return None

    from sculptor.skill_library import SkillLibrary, SkillLibraryHandle

    library = SkillLibrary(root=library_root) if library_root else SkillLibrary()
    return SkillLibraryHandle(
        library=library,
        adapter_class=adapter_class,
        task_id=task_id,
        robot_slug=None,  # CLI doesn't know UI's library_slug; UI may
                          # set this when it grows a Ship 19b surface.
        publish=True,
    )


def _derive_mission_slug(goal: str, existing: set[str]) -> str:
    """Slug derivation mirroring project_store._ensure_unique_slug
    (per Ship 18a plan-review). Conservative slugify + integer suffix
    on collision; never produces empty slugs.

    Audit cross-reference (#C): KEEP IN SYNC with
    `reward-sculptor-ui/backend/services/mission_store._slugify` and
    `derive_unique_mission_slug` — both must produce the same slug
    for the same goal so a CLI-created mission and a REST-created
    mission can be looked up consistently.
    """
    import re as _re
    # ASCII-only snake-case; collapse whitespace + non-word.
    cleaned = _re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")
    # Cap length so the full path stays comfortable.
    base = cleaned[:32].rstrip("-") if cleaned else "mission"
    if not base:
        base = "mission"
    candidate = base
    n = 2
    while candidate in existing:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


@app.command("mission-save")
def mission_save_cli(
    project_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True,
    ),
    mission_slug: str = typer.Argument(
        ..., help="Mission slug under <project_dir>/.missions/."),
    pin: list[str] = typer.Option(
        [], "--pin",
        help="Extra checkpoints to keep, 'stage:iter' (repeatable). "
             "Best + final per stage are kept automatically."),
) -> None:
    """§durable auto-save: archive a mission into the restart-/delete-
    proof `saved/` store (best+final+pinned checkpoints + all videos,
    reports, reward code, metrics). Missions auto-archive as they run;
    use this to (re-)save a pre-existing mission or add pins."""
    from sculptor.archive import archive_mission, saved_root

    mission_dir = (project_dir / ".missions" / mission_slug).resolve()
    if not (mission_dir / "mission.json").is_file():
        typer.echo(f"[mission-save] no mission at {mission_dir}", err=True)
        raise typer.Exit(1)
    pinned: dict[str, set[int]] = {}
    for spec in pin:
        try:
            st, it = spec.split(":")
            pinned.setdefault(st, set()).add(int(it))
        except ValueError:
            typer.echo(f"[mission-save] bad --pin {spec!r} (want stage:iter)",
                       err=True)
            raise typer.Exit(2)
    res = archive_mission(
        mission_dir, saved_root(), project_slug=project_dir.name,
        pinned=pinned or None, incremental=False)
    typer.echo(
        f"[mission-save] archived → {res.entry_dir} "
        f"({int(getattr(res, 'total_bytes', 0) or 0) // (1024*1024)} MB, "
        f"dropped {int(getattr(res, 'dropped_bytes', 0) or 0) // (1024*1024)} MB)")


@app.command("mission-run")
def mission_run_cli(
    project_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True,
    ),
    mission_slug: Optional[str] = typer.Argument(
        None,
        help=(
            "Mission slug under <project_dir>/.missions/. If omitted "
            "AND there is exactly one mission, it's auto-resolved."
        ),
    ),
    no_skill_library: bool = typer.Option(
        False, "--no-skill-library",
        help=(
            "Skip the cross-mission skill library. Default: ON — "
            "successful stages publish to the library; stages with a "
            "Claude-set `init_skill_id` are warm-started from it. "
            "Ship 19."
        ),
    ),
    skill_library_root: Optional[Path] = typer.Option(
        None, "--skill-library-root",
        help=(
            "Override the on-disk root for the skill library. Default: "
            "$SCULPTOR_SKILL_LIBRARY_ROOT or ~/.local/share/sculptor/skills/."
        ),
    ),
    iterations_override: Optional[int] = typer.Option(
        None, "--iterations-override",
        min=1, max=200,
        help=(
            "Override every stage's max_iterations. Use to clamp a long "
            "Claude-authored mission down to a quick smoke (e.g., "
            "--iterations-override 2). Ship 19d."
        ),
    ),
    steps_per_iter: Optional[int] = typer.Option(
        None, "--steps-per-iter",
        min=100, max=200_000,
        help=(
            "Override [iteration].steps_per_iter for every stage. For "
            "mjlab this is the rsl_rl iters per cycle; for gym_sb3 it "
            "is the env-step budget. Useful when stage configs were "
            "scaffolded with the generic 50000 default. Ship 19d."
        ),
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", min=0,
        help="Override the per-iter base seed. Ship 19d.",
    ),
    early_stop_on_criterion: bool = typer.Option(
        False, "--early-stop-on-criterion",
        help=(
            "Goal A (Ship 19d): exit a stage early the moment its "
            "success_criterion holds for `--criterion-stability-window` "
            "consecutive iters. Cuts wall-clock on easy stages. "
            "Default OFF — preserves Ship 16's run-the-full-budget "
            "behavior. Independent of the metric-plateau early-stop "
            "(which already exists at the inner sculpt_run level)."
        ),
    ),
    criterion_stability_window: int = typer.Option(
        1, "--criterion-stability-window",
        min=1, max=10,
        help=(
            "How many consecutive iters the criterion must hold before "
            "Goal A fires. Default 1 (immediate exit on first pass). "
            "Bump to 2-3 for noisier metrics where a single iter could "
            "satisfy the criterion by chance and revert."
        ),
    ),
    extend_on_improvement: bool = typer.Option(
        False, "--extend-on-improvement",
        help=(
            "Goal B (Ship 19d): if a stage finishes its iteration "
            "budget without satisfying the criterion BUT the metric is "
            "still trending up, run additional iters via resume mode. "
            "Default OFF — adaptive extension changes the user's "
            "iteration contract and should be opt-in."
        ),
    ),
    max_extensions_per_stage: int = typer.Option(
        1, "--max-extensions-per-stage",
        min=0, max=3,
        help=(
            "Hard cap on Goal B extensions per stage. Default 1; max 3 "
            "to prevent runaway. Each extension adds "
            "`extension_factor * stage.max_iterations` more iters."
        ),
    ),
    extension_factor: float = typer.Option(
        0.5, "--extension-factor",
        min=0.1, max=1.5,
        help=(
            "Goal B: fraction of original max_iterations to add per "
            "extension. Default 0.5 (e.g. 4-iter stage extends by 2)."
        ),
    ),
    extension_improvement_threshold: float = typer.Option(
        0.05, "--extension-improvement-threshold",
        min=0.0, max=1.0,
        help=(
            "Goal B trend test: recent_best must exceed prior_best by "
            "max(threshold * |prior_best|, 0.05) to count as improving. "
            "Default 5%."
        ),
    ),
    # §MISSION_RUN_PARITY: per-launch knobs mirrored from `sculpt run` so a
    # mission reaches parity with a standalone run. Each applies uniformly
    # to EVERY stage; None = the stage's inherited config value wins.
    edit_candidates: Optional[int] = typer.Option(
        None, "--edit-candidates", min=1, max=5,
        help="Best-of-K framed reward-edit candidates per diagnosis, per "
             "stage (offline-screened; only the winner trains). Injected "
             "into each stage's [iteration].edit_candidates. Omit = 1."),
    rollout_episodes: Optional[int] = typer.Option(
        None, "--rollout-episodes", min=1, max=32,
        help="Rollout episodes captured per iter for behavior metrics, "
             "per stage. Omit = inherited [iteration] value (default 6)."),
    max_episode_steps: Optional[int] = typer.Option(
        None, "--max-episode-steps", min=50, max=5000,
        help="Rollout env steps per episode, per stage. Omit = default 500."),
    playback_speed: Optional[float] = typer.Option(
        None, "--playback-speed", min=0.1, max=10.0,
        help="Rollout video speed multiplier, per stage; 1.0 = real-time."),
    render_width: Optional[int] = typer.Option(
        None, "--render-width",
        help="Rollout video width px, per stage (default 1280)."),
    render_height: Optional[int] = typer.Option(
        None, "--render-height",
        help="Rollout video height px, per stage (default 720)."),
    num_envs: Optional[int] = typer.Option(
        None, "--num-envs", min=1, max=8192,
        help="Override [adapter].config.num_envs for every stage (mjlab). "
             "Drop if a stage OOMs. Omit = inherited value."),
    device: Optional[str] = typer.Option(
        None, "--device",
        help="Override [adapter].config.device for every stage (mjlab), "
             "e.g. cuda:0 / cpu. Omit = inherited value."),
    # §Ship 34: fitness-in-the-loop for every stage (uniform spec metric).
    fitness_metric: Optional[str] = typer.Option(
        None, "--fitness-metric",
        help="Spec-metric name used as ground-truth fitness in EVERY "
             "stage's loop (including capability-driven object_lift_hold). "
             "Sound for single-skill missions. Omit for the blind loop."),
    fitness_target: Optional[float] = typer.Option(
        None, "--fitness-target",
        help="Per-stage: stop once best fitness reaches this (0-1)."),
    fitness_patience: int = typer.Option(
        2, "--fitness-patience", min=1,
        help="Per-stage: stop after N iters with no new best fitness."),
    fitness_mode: str = typer.Option(
        "steer", "--fitness-mode",
        help="'steer' (default): fitness drives per-stage selection/early-"
             "stop. 'observe': compute + display only, no influence."),
    fitness_revert: bool = typer.Option(
        True, "--fitness-revert/--no-fitness-revert",
        help="§Ship 36, per stage (steer mode): on a fitness regression, "
             "revert the edit base to the best-so-far reward instead of "
             "compounding the bad edit. Default on."),
):
    """Run a previously-initialized mission end-to-end.

    Loads `<project_dir>/.missions/<mission_slug>/mission.json`,
    resolves the project's adapter, and calls `mission_run` from
    Ship 16/17 — which iterates stages, materializes v1 from each
    stage's seed prompt, calls sculpt_run with warm-start, evaluates
    success criteria, and re-decomposes on failure.
    """
    from sculptor.adapters.base import load_adapter
    from sculptor.kg.store import SculptorKG
    from sculptor.mission import load_mission
    from sculptor.sculpt import mission_run

    if fitness_mode not in ("steer", "observe"):
        raise typer.BadParameter("--fitness-mode must be 'steer' or 'observe'")

    missions_root = project_dir / ".missions"
    if not missions_root.is_dir():
        typer.echo(
            f"[mission-run] error: no missions in {missions_root}",
            err=True,
        )
        raise typer.Exit(code=2)

    if mission_slug is None:
        slugs = [p.name for p in missions_root.iterdir() if p.is_dir()]
        if len(slugs) == 1:
            mission_slug = slugs[0]
            typer.echo(f"[mission-run] auto-resolved slug: {mission_slug}")
        else:
            typer.echo(
                f"[mission-run] error: {len(slugs)} missions in "
                f"{missions_root}; specify --slug. Found: {slugs}",
                err=True,
            )
            raise typer.Exit(code=2)

    mission_path = missions_root / mission_slug
    if not (mission_path / "mission.json").is_file():
        typer.echo(
            f"[mission-run] error: {mission_path / 'mission.json'} not found",
            err=True,
        )
        raise typer.Exit(code=2)

    mission = load_mission(mission_path)
    # Resolve adapter short-name from project config.
    config_path = project_dir / "config.toml"
    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redef]
        with config_path.open("rb") as f:
            cfg = tomllib.load(f)
        adapter_dotted = ((cfg.get("adapter") or {}).get("class") or "").strip()
    except Exception as e:  # noqa: BLE001
        typer.echo(
            f"[mission-run] error: failed to read {config_path}: {e}",
            err=True,
        )
        raise typer.Exit(code=2) from e

    # Reverse-resolve dotted-path → short-name. Mirrors sculpt.py's
    # `_ADAPTER_SHORT_NAMES` mapping; falls back to the dotted name.
    from sculptor.sculpt import _ADAPTER_SHORT_NAMES
    short_name = next(
        (k for k, v in _ADAPTER_SHORT_NAMES.items() if v == adapter_dotted),
        adapter_dotted,
    )

    # §Ship 19: build a skill-library handle from the project's
    # adapter + task_id, unless `--no-skill-library` was passed.
    handle = (
        None if no_skill_library
        else _build_skill_library_handle(
            config_path, library_root=skill_library_root,
        )
    )

    _warn_fitness_metric_mismatch(config_path, fitness_metric)

    kg_store = SculptorKG()
    try:
        from sculptor.world.channels import load_project_channel_catalog

        result = mission_run(
            mission,
            adapter_short_name=short_name,
            kg_store=kg_store,
            skill_library_handle=handle,
            iterations_override=iterations_override,
            steps_per_iter=steps_per_iter,
            seed=seed,
            early_stop_on_criterion=early_stop_on_criterion,
            criterion_stability_window=criterion_stability_window,
            extend_on_improvement=extend_on_improvement,
            max_extensions_per_stage=max_extensions_per_stage,
            extension_factor=extension_factor,
            extension_improvement_threshold=extension_improvement_threshold,
            fitness_metric=fitness_metric,
            fitness_target=fitness_target,
            fitness_patience=fitness_patience,
            fitness_observe_only=(fitness_mode == "observe"),
            fitness_revert=fitness_revert,
            # §MISSION_RUN_PARITY: per-launch knobs → every stage.
            edit_candidates=edit_candidates,
            rollout_episodes=rollout_episodes,
            max_episode_steps=max_episode_steps,
            playback_speed=playback_speed,
            render_width=render_width,
            render_height=render_height,
            num_envs=num_envs,
            device=device,
            channel_catalog=load_project_channel_catalog(project_dir),
        )
    finally:
        kg_store.close()

    typer.echo(
        f"[mission-run] completed={result.completed}; "
        f"halted_at={result.halted_at_stage}; "
        f"reason={result.halted_reason}"
    )
    if not result.completed:
        raise typer.Exit(code=1)


@app.command()
def report(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, readable=True,
        help="Project config.toml. Report is written into its directory."),
    out: Path = typer.Option(
        Path("final.mp4"), "--out", "-o",
        help="Output path for the time-lapse mp4."),
):
    """Build the final report: `final.mp4` + `<project>/reports/final_report.md`."""
    from sculptor.timelapse import build_report

    result = build_report(config_path=config, out_mp4=out)
    typer.echo(f"[report] wrote {result.final_report_md_path}")
    if result.final_mp4_ok:
        typer.echo(f"[report] wrote {result.final_mp4_path}")
    else:
        typer.echo(
            f"[report] mp4 build failed or produced empty file at "
            f"{result.final_mp4_path}")
        if result.ffmpeg_stderr:
            typer.echo(
                "[report] ffmpeg stderr:\n" + result.ffmpeg_stderr[-800:])
    if result.selected_iter_indices:
        typer.echo(
            f"[report] panels from iters {result.selected_iter_indices}")


@app.command()
def export(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, readable=True,
        help="Project config.toml. The bundle lands in <project>/exports/."),
    iter_index: Optional[int] = typer.Option(
        None, "--iter", "-i",
        help="Iteration to export (default: latest with a checkpoint)."),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o",
        help="Output zip path (default: <project>/exports/policy_<name>_iter<N>.zip)."),
    runs_root: Optional[Path] = typer.Option(
        None, "--runs-root",
        help="Alternate runs/ tree, e.g. a mission stage's "
             ".missions/<m>/stages/<s>/runs (default: <project>/runs)."),
    list_only: bool = typer.Option(
        False, "--list", help="List exportable iterations and exit."),
):
    """Export a trained policy as a self-contained deployment bundle.

    The zip contains the raw checkpoint, best-effort ONNX + TorchScript
    exports of the actor network, the exact reward version + env spec the
    iteration trained under, the project config, metrics, and a DEPLOY.md
    loading recipe — everything a sim-to-real pipeline needs in one file.
    """
    from sculptor.export import (
        ExportError,
        export_policy_bundle,
        list_exportable_iters,
    )

    project = config.resolve().parent
    root = runs_root if runs_root is not None else project / "runs"

    if list_only:
        rows = list_exportable_iters(root)
        if not rows:
            typer.echo(f"[export] no exportable iterations under {root}")
            raise typer.Exit(1)
        for r in rows:
            metric = (
                f"{r['primary_metric']:.2f}"
                if r["primary_metric"] is not None else "—")
            typer.echo(
                f"  iter {r['iter_index']:>3}  {r['checkpoint']:<14} "
                f"reward={r['reward_version'] or '—':<5} metric={metric}")
        return

    try:
        result = export_policy_bundle(
            project, iter_index=iter_index, runs_root=root, out_path=out)
    except (ExportError, OSError) as e:
        typer.echo(f"[export] {e}", err=True)
        raise typer.Exit(1)
    net = result.manifest.get("network") or {}
    typer.echo(f"[export] wrote {result.bundle_path}")
    if net.get("exports"):
        typer.echo(f"[export] network exports: {', '.join(net['exports'])}")
    for w in result.warnings:
        typer.echo(f"[export] warning: {w}", err=True)


@app.command()
def heldout(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, readable=True,
        help="Project config.toml. Report lands in <project>/reports/heldout/."),
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", exists=True, readable=True,
        help="Trained checkpoint to evaluate (a kept-best policy)."),
    metric: str = typer.Option(
        ..., "--metric",
        help="HAND spec-metric name (e.g. g1_jump). Generated metrics are "
             "rejected — the battery exists to score OUTSIDE the loop."),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Output dir (default <project>/reports/heldout)."),
    push_levels: str = typer.Option(
        "0,0.5,1.0,1.5", "--push-levels",
        help="Comma-separated push magnitudes m/s (0 = unperturbed base)."),
    seeds: str = typer.Option(
        "70001,70002,70003", "--seeds",
        help="Fresh rollout seeds (70k band — disjoint from the loop's "
             "selection seeds by convention)."),
    episodes: int = typer.Option(3, "--episodes"),
):
    """§RESEARCH_GAP_ANALYSIS §7.4: held-out evaluation battery.

    Scores a kept policy on a push-perturbation grid over the frozen
    shared env, with fresh seeds and hand spec metrics — numbers the
    loop never optimized. Writes `heldout_report.json`.
    """
    import json as _json

    from sculptor.adapters.base import load_adapter
    from sculptor.env_spec import read_current_env_spec
    from sculptor.eval.heldout import run_heldout_battery

    project = config.resolve().parent
    adapter = load_adapter(config)
    base_spec = read_current_env_spec(project / "env")
    out_dir = out if out is not None else project / "reports" / "heldout"
    report = run_heldout_battery(
        adapter=adapter,
        checkpoint_path=checkpoint,
        out_dir=out_dir,
        metric=metric,
        base_env_spec=base_spec,
        push_levels=[float(x) for x in push_levels.split(",") if x.strip()],
        seeds=[int(x) for x in seeds.split(",") if x.strip()],
        n_episodes=episodes,
        on_event=lambda ev: typer.echo(f"[heldout] {_json.dumps(ev)}"),
    )
    typer.echo(f"[heldout] report: {out_dir / 'heldout_report.json'}")
    for row in report["levels"]:
        typer.echo(
            f"  push {row['push_mps']:>4g} m/s  "
            f"median {row['median_score']:.3f}  "
            f"degradation {row['degradation_vs_base']}")
    if not report["env_spec_lever_available"]:
        typer.echo(
            "[heldout] WARNING: adapter has no env_spec lever — perturbed "
            "cells ran UNPERTURBED (marked in report).", err=True)


# ── sculpt reference: RSI curricula from reference clips ─────────────────
reference_app = typer.Typer(
    help="Reference trajectories: derive RSI train-curricula from motion "
         "clips (DeepMimic RSI; train-only, rollout evaluation untouched).")
app.add_typer(reference_app, name="reference")


@reference_app.command("jump")
def reference_jump(
    project: Path = typer.Option(
        ..., "--project",
        help="Project dir (clip → <project>/reference/, spec → <project>/env/)."),
    stand_height: float = typer.Option(
        0.78, "--stand-height", help="Standing base height in metres "
        "(0.78 = Unitree G1)."),
    apex_gain: float = typer.Option(
        0.35, "--apex", help="Jump apex above standing, metres."),
    crouch_frac: float = typer.Option(
        0.62, "--crouch", help="Crouch depth as a fraction of stand."),
    clip: Optional[Path] = typer.Option(
        None, "--clip",
        help="Existing clip .npz (e.g. converted retargeted mocap) instead "
             "of the procedural jump."),
    apply: bool = typer.Option(
        True, "--apply/--no-apply",
        help="Persist the derived RSI curriculum as the next validated "
             "env-spec version (train scope only)."),
) -> None:
    """Generate (or load) a jump reference clip, print its measured phase
    keyframes (crouch depth / takeoff vz / apex / flight time — prompt-
    ready numbers instead of guessed thresholds), and derive a validated
    RSI train-curriculum from its airborne states."""
    import json as _json

    from sculptor.reference import (
        apply_reference_rsi, derive_rsi_train_keys, load_clip,
        make_procedural_jump_clip, phase_keyframes, save_clip)

    project = project.resolve()
    if clip is not None:
        c = load_clip(clip)
        typer.echo(f"[reference] loaded clip: {clip}")
    else:
        c = make_procedural_jump_clip(
            stand_height_m=stand_height, apex_gain_m=apex_gain,
            crouch_frac=crouch_frac)
        out = save_clip(project / "reference" / "jump.npz", c)
        typer.echo(f"[reference] clip written: {out}")
    typer.echo(_json.dumps(phase_keyframes(c), indent=2))
    if apply:
        path = apply_reference_rsi(project / "env", c)
        typer.echo(
            f"[reference] env spec written: {path} (train-only RSI + paired "
            "sunk termination; rollout evaluation untouched)")
    else:
        typer.echo("[reference] derived train keys (not applied):")
        typer.echo(_json.dumps(derive_rsi_train_keys(c), indent=2))


# ── sculpt refs: reference motion library (§R1_BUILD_SPEC) ──────────────
refs_app = typer.Typer(
    name="refs",
    help="Reference motion library: ingest/index/list retargeted mocap "
         "clips (LAFAN1-g1, fleaven-g1) for RSI. `refs search` lands in a "
         "later worker.",
    no_args_is_help=True,
)
app.add_typer(refs_app, name="refs")


@refs_app.command("ingest")
def refs_ingest(
    source: str = typer.Option(
        ..., "--source",
        help="Dataset source: lafan1-g1 | fleaven-g1."),
    filter_glob: Optional[str] = typer.Option(
        None, "--filter", help="fnmatch glob over source filenames, "
        "e.g. '*fallAndGetUp*'."),
    no_preview: bool = typer.Option(
        False, "--no-preview",
        help="Skip preview.png generation (also the automatic fallback "
             "when the preview module/GL context is unavailable)."),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Cap number of source files fetched this run."),
    all_: bool = typer.Option(
        False, "--all",
        help="fleaven-g1 only: walk the FULL g1/**/*.npy tree (every "
             "page of the HF tree API) instead of the default single-"
             "page listing. Default off — without this flag, behavior "
             "is unchanged from before this flag existed."),
    manifest_out: Optional[Path] = typer.Option(
        None, "--manifest-out",
        help="With --all: write/reuse the enumerated file list (path+"
             "size) as JSON at this path, so a full-tree run can be "
             "resumed/audited. If the file exists and is < 1 day old, "
             "it is reused instead of re-enumerating (see "
             "--refresh-manifest)."),
    refresh_manifest: bool = typer.Option(
        False, "--refresh-manifest",
        help="With --all --manifest-out: force re-enumeration even if "
             "an existing manifest at that path looks fresh."),
) -> None:
    """Download + validate + index a batch of clips from a public HF
    dataset (plain HTTPS, ungated). Idempotent: re-running skips clips
    whose content hash is already indexed. Rejects are never fatal —
    logged to `index_rejects.jsonl` with a reason; see that file for
    anything skipped this run. Unless `--no-preview`, a best-effort
    `preview.png` keyframe strip is rendered per accepted clip/segment
    (§decision 8) — a missing preview module or GL/EGL context is
    logged per-clip and never fails the ingest."""
    from sculptor.refs.ingest import ingest_source
    from sculptor.refs.library import rebuild_index

    summary = ingest_source(
        source, filter_glob=filter_glob, limit=limit, no_preview=no_preview,
        full_tree=all_, manifest_path=manifest_out,
        refresh_manifest=refresh_manifest,
        progress=lambda msg: typer.echo(msg))

    rows = rebuild_index()
    typer.echo(
        f"[refs ingest] accepted={len(summary.accepted)} "
        f"rejected={len(summary.rejected)} "
        f"skipped_existing={len(summary.skipped_existing)} "
        f"index_rows={len(rows)}")
    if summary.rejected:
        typer.echo("[refs ingest] rejected clips:")
        for clip_id, reason in summary.rejected:
            typer.echo(f"  - {clip_id}: {reason}")


@refs_app.command("index")
def refs_index() -> None:
    """Rebuild `index.jsonl` from every `<robot>/<clip_id>/provenance.json`
    on disk. The index is a cache — provenance is truth — so this is safe
    to run any time (e.g. after a manual edit or a partial ingest)."""
    from sculptor.refs.library import rebuild_index

    rows = rebuild_index()
    typer.echo(f"[refs index] rebuilt {len(rows)} row(s)")


@refs_app.command("list")
def refs_list(
    robot: Optional[str] = typer.Option(
        None, "--robot", help="Filter to one robot (default: all)."),
) -> None:
    """List indexed clips (reads index.jsonl; run `sculpt refs index` first
    if it's missing or stale)."""
    from sculptor.refs.library import read_index

    rows = read_index()
    if robot:
        rows = [r for r in rows if r.get("robot") == robot]
    if not rows:
        typer.echo("[refs list] no clips indexed (run `sculpt refs ingest` "
                    "then `sculpt refs index`)")
        return
    for row in rows:
        preview = "preview" if row.get("has_preview") else "no-preview"
        typer.echo(
            f"{row['clip_id']:<40} robot={row['robot']:<6} "
            f"tier={row.get('tier', '?'):<3} "
            f"frames={row.get('n_frames', '?'):<6} "
            f"fps={row.get('fps', '?'):<6} "
            f"dur={row.get('duration_s', '?')}s "
            f"[{preview}]  {row.get('text', '')}")


@refs_app.command("search")
def refs_search(
    query: str = typer.Argument(..., help="Free-text goal, e.g. "
        "'get up off the ground'."),
    robot: str = typer.Option("g1", "--robot", help="Robot to search."),
    k: int = typer.Option(10, "--k", help="Max results to print."),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Skip the optional LLM rerank layer "
        "(deterministic token-overlap ranking only)."),
) -> None:
    """Rank indexed clips against a free-text query (§decision 7).
    Deterministic token-overlap + synonym-expanded scoring always runs;
    unless `--no-llm`, the top candidates are reranked by
    `reference_rerank` with a match_confidence + reason — any failure
    there (no key, network, parse) silently falls back to the
    deterministic ranking, never raises."""
    from sculptor.refs.retrieve import search

    results = search(query, robot=robot, k=k, use_llm=not no_llm)
    if not results:
        typer.echo(f"[refs search] no matches for {query!r} (robot={robot})")
        return
    for m in results:
        conf = f"{m.match_confidence:.2f}" if m.match_confidence is not None else "  - "
        typer.echo(
            f"{m.clip_id:<40} score={m.score:>8.3f} conf={conf} "
            f"tier={m.tier or '?':<3} dur={m.duration_s or '?'}s "
            f"[{m.rerank}]  {m.text}")
        if m.reason:
            typer.echo(f"    reason: {m.reason}")


@refs_app.command("preview")
def refs_preview(
    clip_id: str = typer.Argument(..., help="Clip id to render/re-render."),
    robot: str = typer.Option("g1", "--robot", help="Robot the clip belongs to."),
) -> None:
    """Render (or re-render) a single clip's `preview.png` keyframe
    strip on demand. Skips cleanly (non-zero exit, actionable message)
    if `sculptor.refs.preview` can't create a GL/EGL context in this
    environment — never a stack trace."""
    from sculptor.reference import load_clip
    from sculptor.refs import library
    from sculptor.refs.preview import (
        PreviewUnavailable, render_preview_png, resolve_mjcf_for_robot)

    clip_path = library.clip_dir(robot, clip_id) / library.CLIP_FILENAME
    if not clip_path.is_file():
        typer.echo(f"[refs preview] no such clip: {robot}/{clip_id}", err=True)
        raise typer.Exit(code=1)
    clip = load_clip(clip_path)
    out_path = library.clip_dir(robot, clip_id) / library.PREVIEW_FILENAME
    try:
        mjcf_path = resolve_mjcf_for_robot(robot)
        render_preview_png(clip, out_path, mjcf_path=mjcf_path)
    except PreviewUnavailable as e:
        typer.echo(f"[refs preview] unavailable in this environment: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"[refs preview] wrote {out_path}")


@refs_app.command("retarget")
def refs_retarget(
    source: Path = typer.Option(
        ..., "--source", help="Motion source file (BVH or SMPL-X npz)."),
    format_: str = typer.Option(
        "bvh", "--format", help="Source format: bvh | smplx."),
    robot: list[str] = typer.Option(
        ..., "--robot",
        help="Target robot library slug (repeatable), e.g. --robot g1 "
             "--robot t1. See sculptor.refs.retarget.GMR_ROBOT_IDS for "
             "the supported slugs."),
    bvh_format: str = typer.Option(
        "lafan1", "--bvh-format",
        help="BVH bone-naming convention: lafan1 | nokov (bvh source only)."),
    license_: str = typer.Option(
        ..., "--license", help="License tag for the source clip (provenance)."),
    attribution: str = typer.Option(
        ..., "--attribution", help="Attribution string for the source clip."),
    text: str = typer.Option("", "--text", help="Free-text label for retrieval."),
    labels: Optional[str] = typer.Option(
        None, "--labels", help="Comma-separated labels."),
    roles: Optional[str] = typer.Option(
        None, "--roles",
        help="Comma-separated joint roles to verify resolution for (e.g. "
             "'left_hip_pitch,right_hip_pitch'). Skipped if omitted."),
    gmr_python: Optional[Path] = typer.Option(
        None, "--gmr-python",
        help="Path to GMR's venv python (default: ~/tools/GMR/.venv/bin/python)."),
) -> None:
    """Retarget ONE source motion clip to one or more robots via GMR
    (cross-venv subprocess — see sculptor.refs.retarget), registering
    each result in the reference library with retarget provenance. Also
    attempts a best-effort preview render per clip — MJCF resolved BY
    ROBOT via sculptor.refs.preview.resolve_mjcf_for_robot (g1 from the
    installed mjlab package, t1 from a local GMR checkout's own asset
    tree; a robot with no registered resolver logs a skip and never
    fails the run — the clip stays valid with no preview.png)."""
    from sculptor.refs.retarget import (
        RetargetError, attach_role_resolution_qc, retarget_and_register)
    from sculptor.refs import library

    label_list = [s.strip() for s in labels.split(",") if s.strip()] if labels else []
    role_list = [s.strip() for s in roles.split(",") if s.strip()] if roles else []

    for r in robot:
        typer.echo(f"[refs retarget] {source} -> robot={r} (format={format_})")
        try:
            lc = retarget_and_register(
                source, format_, r,
                license_=license_, attribution=attribution, text=text,
                labels=label_list, gmr_python=gmr_python, bvh_format=bvh_format)
        except RetargetError as e:
            typer.echo(f"[refs retarget] FAILED for robot={r}: {e}", err=True)
            continue
        typer.echo(
            f"[refs retarget] registered {lc.clip_id} "
            f"(robot={lc.robot}, npz={lc.clip_path})")

        if role_list:
            summary = attach_role_resolution_qc(r, lc.clip_id, role_list)
            status = "OK" if summary["ok"] else "FAILED"
            typer.echo(f"[refs retarget] role resolution {status}: {summary}")

        try:
            from sculptor.reference import load_clip
            from sculptor.refs.preview import (
                PreviewUnavailable, render_preview_png, resolve_mjcf_for_robot)

            clip = load_clip(lc.clip_path)
            out_path = library.clip_dir(r, lc.clip_id) / library.PREVIEW_FILENAME
            mjcf_path = resolve_mjcf_for_robot(r)
            render_preview_png(clip, out_path, mjcf_path=mjcf_path)
            typer.echo(f"[refs retarget] preview written: {out_path}")
        except PreviewUnavailable as e:
            typer.echo(f"[refs retarget] preview unavailable for robot={r}: {e}")
        except Exception as e:  # noqa: BLE001 — preview must never fail the run
            typer.echo(
                f"[refs retarget] preview skipped for robot={r}: "
                f"{type(e).__name__}: {e}")

    rows = library.rebuild_index()
    typer.echo(f"[refs retarget] index_rows={len(rows)}")


@refs_app.command("resegment")
def refs_resegment(
    parent: str = typer.Option(
        ..., "--parent", help="clip_id of the parent clip to re-segment "
        "(the un-suffixed clip, not one of its `--segNN` children)."),
    robot: str = typer.Option("g1", "--robot", help="Robot the parent clip belongs to."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would change without "
        "writing or deleting anything."),
    no_preview: bool = typer.Option(
        False, "--no-preview", help="Skip preview.png re-render for the "
        "new segments."),
) -> None:
    """Re-run segmentation for one already-indexed parent clip using the
    current `sculptor.refs.segment` rules (2026-07-09 settled-start
    fix), replacing its existing derived `--segNN` segments. Only clips
    whose provenance `parent_clip_id` matches `--parent` are touched —
    the rest of the library is untouched. QC-rejected candidates are
    logged (never written) via the same rejects mechanism ingest uses."""
    from sculptor.refs.ingest import ResegmentError, resegment_clip

    try:
        summary = resegment_clip(
            parent, robot=robot, dry_run=dry_run, no_preview=no_preview,
            progress=lambda msg: typer.echo(msg))
    except ResegmentError as e:
        typer.echo(f"[refs resegment] FAILED: {e}", err=True)
        raise typer.Exit(code=1) from e

    verb = "would remove" if dry_run else "removed"
    add_verb = "would add" if dry_run else "added"
    typer.echo(
        f"[refs resegment] parent={summary.parent_clip_id} "
        f"{verb}={len(summary.removed)} {add_verb}={len(summary.added)} "
        f"rejected={len(summary.rejected)}")
    for seg_id in summary.removed:
        typer.echo(f"  - {verb}: {seg_id}")
    for seg_id in summary.added:
        typer.echo(f"  + {add_verb}: {seg_id}")
    for cand_id, reason in summary.rejected:
        typer.echo(f"  x rejected: {cand_id}: {reason}")


@refs_app.command("track")
def refs_track(
    clip_id: str = typer.Option(
        ..., "--clip-id", help="Tier-K clip_id to certify to Tier D."),
    robot: str = typer.Option("g1", "--robot", help="Robot the clip belongs to."),
    donor_project: Path = typer.Option(
        ..., "--donor-project",
        help="Path to an existing sculpt project whose config.toml "
             "[adapter] table (class + config) is templated into the "
             "throwaway tracking project."),
    iterations: int = typer.Option(
        3, "--iterations", help="Number of adapter.train() calls "
        "(each warm-started from the prior checkpoint)."),
    steps_per_iteration: int = typer.Option(
        2000, "--steps-per-iteration", help="mjlab max_iterations per "
        "adapter.train() call (see MjlabAdapter.train's docstring: "
        "'steps' IS max_iterations, not raw env steps)."),
    n_episodes: int = typer.Option(
        2, "--n-episodes", help="Rollout episodes scored against the clip."),
    seed: int = typer.Option(0, "--seed", help="Train/rollout seed."),
    project_dir: Optional[Path] = typer.Option(
        None, "--project-dir",
        help="Throwaway project directory (default: "
             "<clip_dir>/tierD_work)."),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Build the throwaway project (config + tracking reward + "
             "RSI/eval-reset env spec) and print the plan without "
             "training."),
) -> None:
    """Tier-D certification (§REFERENCE_TRAJECTORY_PLAN §2.3, §11 R4):
    physics-track a Tier-K clip in our own mjlab sim with a bounded
    DeepMimic-style tracking run. Success within tolerance upgrades the
    clip's provenance tier K -> D and copies the tracked rollout beside
    the clip as `tierD_rollout.npz`; failure records
    `tierD.feasible=false` (a useful verdict, not an error) and leaves
    the tier unchanged. See `sculptor.refs.track` for the full pipeline."""
    import json as _json

    from sculptor.refs.track import TrackError, track_clip

    try:
        result = track_clip(
            clip_id=clip_id, robot=robot, donor_project=donor_project,
            iterations=iterations, steps_per_iteration=steps_per_iteration,
            n_episodes=n_episodes, seed=seed, project_dir=project_dir,
            dry_run=dry_run, progress=lambda msg: typer.echo(msg))
    except TrackError as e:
        typer.echo(f"[refs track] FAILED: {e}", err=True)
        raise typer.Exit(code=1) from e

    if result.dry_run:
        typer.echo(
            f"[refs track] dry-run plan: project_dir={result.plan.project_dir} "
            f"reward={result.plan.reward_path} config={result.plan.config_path} "
            f"iterations={result.plan.iterations} "
            f"steps_per_iteration={result.plan.steps_per_iteration} "
            f"n_episodes={result.plan.n_episodes} "
            f"joint_names={result.plan.joint_names}")
        return

    assert result.errors is not None
    verdict = "FEASIBLE (tier -> D)" if result.errors.feasible else "INFEASIBLE (tier stays K)"
    typer.echo(f"[refs track] {clip_id}: {verdict}")
    typer.echo(_json.dumps(result.errors.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
