"""sculptor/cli.py — `sculpt` command-line entry point.

Phase 1 scaffolding for top-level commands (`init`, `run`, `resume`, `viz`) —
they print "not implemented yet" until the inner loop lands. The `kg`
subcommand group (list-papers, list-techniques, stats) is wired to the real
store (see `sculptor.kg.store.SculptorKG`).
"""

from __future__ import annotations

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


_STORE_OPT = typer.Option(
    None,
    "--store",
    help="Path to the KG DB (default: $SCULPTOR_KG_PATH or ./kg/graph.db).",
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
    print_one: bool = typer.Option(
        True, "--print-one/--no-print-one",
        help="Dump the first successful payload as JSON for inspection."),
    store: Optional[Path] = _STORE_OPT,
):
    """Run LLM extraction over Papers in the KG.

    Requires `ANTHROPIC_API_KEY` in the environment. Creates Technique,
    FailureMode, RewardComponent, Environment nodes and their edges.
    """
    from sculptor.kg.extract import cli_extract_all

    if not (all_ or force):
        typer.echo("specify --all to extract every unextracted paper")
        raise typer.Exit(code=2)
    raise typer.Exit(code=cli_extract_all(store, force=force, limit=limit, print_one=print_one))


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
        help="§Ship-9a: enable or disable the early-stop check. "
             "Disable for long overnight runs where a transient metric "
             "dip may mask real behavioral improvement."),
    early_stop_patience: Optional[int] = typer.Option(
        None, "--early-stop-patience",
        help="§Ship-9a: consecutive iterations with no primary_metric "
             "improvement before early-stop fires (default 3)."),
):
    """Run the inner loop: train → rollout → diagnose → edit → commit."""
    from sculptor.sculpt import sculpt_run

    result = sculpt_run(
        config_path=config, behavior_goal=behavior, iterations=iterations,
        resume=resume_run, no_kg=no_kg, dry_run=dry_run,
        steps_per_iter=steps_per_iter,
        max_episode_steps=max_episode_steps,
        playback_speed=playback_speed,
        render_every=render_every,
        rollout_fps=rollout_fps,
        rollout_episodes=rollout_episodes,
        seed=seed,
        auto_adjust_physics=auto_adjust_physics,
        early_stop_enabled=early_stop_enabled,
        early_stop_patience=early_stop_patience,
    )
    typer.echo(
        f"[sculpt] done. iters_run={result.iterations_run} "
        f"early_stopped={result.early_stopped}")
    if result.early_stopped:
        typer.echo(f"[sculpt] {result.early_stop_reason}")
    if result.final_reward_path:
        typer.echo(f"[sculpt] final reward: {result.final_reward_path}")


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

    mission_dir = missions_root / mission_slug
    mission.mission_dir = str(mission_dir.resolve())
    save_mission(mission, mission_dir)

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

    kg_store = SculptorKG()
    try:
        result = mission_run(
            mission,
            adapter_short_name=short_name,
            kg_store=kg_store,
            skill_library_handle=handle,
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


if __name__ == "__main__":  # pragma: no cover
    app()
