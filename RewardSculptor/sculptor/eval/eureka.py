"""Eureka-style baseline condition (§Ship 28 / E3).

Faithful-enough reimplementation of Eureka (Ma et al., 2023,
arXiv:2310.12931): per generation, sample K reward-function candidates
from the LLM, train each independently, select by task fitness, feed a
"reward reflection" of the best candidate into the next generation.
No knowledge graph, no diagnosis, no curriculum — generate → train →
select → reflect.

Documented deltas from the original (methodology review requires these
be explicit, not discovered):

  1. **LLM**: Claude (the same model id the sculptor pipeline uses)
     instead of GPT-4 — holding the LLM constant across conditions is
     what makes the comparison about the METHOD.
  2. **Task fitness F**: our hand-authored spec metric. Eureka assumes
     ground-truth fitness access for candidate selection and
     reflection; the spec metric is our F. NOTE the asymmetry: the
     sculptor conditions NEVER see the spec metric (they navigate by
     their own LLM-authored criteria) while this baseline selects on
     it directly — an advantage GIVEN TO THE BASELINE, i.e.
     conservative with respect to the project's hypothesis.
  3. **Environment context**: Eureka pastes the env's source code into
     the prompt; mjlab envs aren't a single source file, so candidates
     get the same state schema / info-key contract the sculptor's own
     edit prompts receive (equal information across conditions).
  4. **Sampling**: K sequential samples at temperature 1.0 (Eureka's
     sampling temperature — note the Anthropic SDK default is also
     1.0, so this is documentation, not a config asymmetry) instead of
     one n=K batched call. Within a job every candidate trains on the
     SAME seed (training stochasticity held constant so fitness
     differences are reward-driven — matches Eureka practice; the
     search can overfit that one seed, as in the original).
  5. **Invalid candidates**: validated by a static
     `compute_reward_batched` presence check + the zero-tensor probe;
     one resample, then scored 0 for the generation (Eureka assigns
     -inf and regenerates next generation; an honest zero keeps the
     accounting visible). LLM API failures count as invalid candidates
     with the error recorded — never a crashed job.
  6. **Deliberation budget**: candidates are sampled WITH adaptive
     thinking and the same max_tokens as the sculptor's edit calls —
     the original used non-thinking GPT-4, but handicapping the
     baseline's inference budget relative to the pipeline would be the
     unfair direction. A strengthening deviation, conservative w.r.t.
     the project's hypothesis.
  7. **Reflection target**: the best candidate ACROSS generations in a
     fresh single-turn prompt each generation, vs the paper's
     accumulating conversation reflecting on the CURRENT iteration's
     best. Never mutates away from a regression — favors the baseline.
     The reflection carries the per-component training SERIES
     (10 downsampled points + max/mean/min — the same Appendix-F shape
     the sculptor's own edit prompt receives) plus the spec components
     of the best candidate; exposing the spec's component values leaks
     more fitness structure than the original's scalar F — again, a
     baseline-favoring asymmetry.

Scoring: a Eureka job's headline `final_spec_score` is the best across
generations — that IS the method's defined output (Algorithm 1 returns
the best reward over all iterations); scoring its last generation would
strawman it. GPU accounting: a job trains up to `generations × K`
candidates — `total_rl_iterations` bills the candidates that actually
trained (budget attributable to recorded results), and campaign reports
must be read against it (Ship-27 fairness machinery).
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from sculptor.prompts import load_prompt

MODEL_ID = "claude-opus-4-7"
#: Eureka samples at temperature 1.0 for candidate diversity (the SDK
#: default is also 1.0; recorded via the TEMPERATURE attr convention).
TEMPERATURE = 1.0
#: Matches sculptor.edit's budget — see documented delta 6.
MAX_TOKENS = 16000

_FENCE_RE = re.compile(r"```(?:[A-Za-z]+)?\s*\n(.*?)```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """The candidate must be a python module. When the model split the
    module across several fenced blocks, join the code-bearing ones (a
    block defining functions/REWARD_SPEC is module content; prose
    blocks aren't); otherwise take the largest block; fall back to the
    raw text when the model skipped fences."""
    blocks = _FENCE_RE.findall(text or "")
    if blocks:
        code_blocks = [
            b for b in blocks
            if "def " in b or "REWARD_SPEC" in b
        ]
        if len(code_blocks) > 1:
            return "\n\n".join(b.strip() for b in code_blocks) + "\n"
        if code_blocks:
            return code_blocks[0].strip() + "\n"
        return max(blocks, key=len).strip() + "\n"
    return (text or "").strip() + "\n"


def _sample_candidate(
    client: Any,
    system_prompt: str,
    user_content: str,
    *,
    model: str = MODEL_ID,
    temperature: float = TEMPERATURE,
) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        # Delta 6: same deliberation budget as the pipeline's edit
        # calls — handicapping the baseline would be the unfair
        # direction.
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    chunks = [
        block.text for block in resp.content
        if getattr(block, "type", "") == "text"
    ]
    return _strip_code_fences("\n".join(chunks))


def _statically_valid(source: str) -> Optional[str]:
    """Cheap pre-probe gate: the probe exercises only the scalar path,
    so a candidate missing `compute_reward_batched` would pass it and
    die at train time — spend the resample budget here instead."""
    for required in ("REWARD_SPEC", "def compute_reward",
                     "def compute_reward_batched"):
        if required not in source:
            return f"candidate source lacks {required!r}"
    return None


def _downsample(vals: list[float], n_points: int = 10) -> dict[str, Any]:
    """Appendix-F-shaped summary: n evenly spaced points + max/mean/min
    — the SAME format the sculptor's edit prompt receives, so neither
    condition has a feedback-information edge."""
    if not vals:
        return {"points": [], "max": None, "mean": None, "min": None}
    if len(vals) <= n_points:
        pts = [round(v, 5) for v in vals]
    else:
        idx = [round(i * (len(vals) - 1) / (n_points - 1))
               for i in range(n_points)]
        pts = [round(vals[i], 5) for i in idx]
    return {
        "points": pts,
        "max": round(max(vals), 5),
        "mean": round(sum(vals) / len(vals), 5),
        "min": round(min(vals), 5),
    }


def _reward_reflection(
    best: Optional[dict[str, Any]], iter_dir: Optional[Path],
) -> dict[str, Any]:
    """Eureka's 'reward reflection': the best candidate's fitness plus
    per-component training SERIES (downsampled points + max/mean/min,
    same shape as the original's checkpoint statistics and as the
    sculptor's own edit feedback), so the next generation can see WHICH
    terms saturated or collapsed."""
    if best is None:
        return {"note": "first generation — no prior candidates"}
    reflection: dict[str, Any] = {
        "fitness_spec_score": best.get("fitness"),
        "spec_components": {
            k: v for k, v in (best.get("spec") or {}).items()
            if isinstance(v, (int, float))
        },
    }
    if iter_dir is None:
        # Champion trained but its artifacts are gone/incomplete
        # (rollout crash) — fitness-only reflection, never the
        # contradictory "no prior candidates" note next to a source.
        reflection["note"] = (
            "component training series unavailable for the best candidate"
        )
        return reflection
    traj_path = iter_dir / "reward_trajectory.json"
    if traj_path.is_file():
        try:
            traj = json.loads(traj_path.read_text(encoding="utf-8"))
            reflection["reward_components_over_training"] = {
                name: _downsample(list(vals))
                for name, vals in traj.items() if vals
            }
        except Exception:  # noqa: BLE001 — reflection is best-effort
            pass
    return reflection


def run_eureka_job(
    *,
    config_path: Path,
    job_dir: Path,
    behavior_goal: str,
    spec_metric: str,
    generations: int,
    k: int,
    steps_per_iter: int,
    rollout_episodes: int,
    seed: int,
    client: Any = None,
    model: str = MODEL_ID,
) -> dict[str, Any]:
    """Run one Eureka-style job. Mirrors each generation's BEST
    candidate into `runs/iter_<g>/` so the harness's standard spec
    series / iterations-to-threshold machinery reads it unchanged.
    Returns a summary {generations, candidates_sampled,
    candidates_trained} and writes the full per-candidate audit trail
    to `eureka_log.json`."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=2, timeout=240.0)

    from sculptor.adapters.base import load_adapter
    from sculptor.eval.spec_metrics import compute_spec_metrics

    adapter = load_adapter(config_path)
    contract = adapter.reward_contract()
    system_prompt = load_prompt("eureka_baseline")

    log: dict[str, Any] = {"generations": [], "model": model,
                           "temperature": TEMPERATURE, "k": k}
    best_overall: Optional[dict[str, Any]] = None
    best_overall_iter: Optional[Path] = None
    candidates_sampled = 0
    candidates_trained = 0

    for g in range(generations):
        gen_dir = job_dir / "eureka" / f"gen_{g}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        reflection = _reward_reflection(best_overall, best_overall_iter)
        user_content = json.dumps({
            "task_description": behavior_goal,
            "generation": g,
            "state_schema": {
                name: list(shape)
                for name, shape in (contract.state_schema or {}).items()
            },
            "info_keys": list(contract.expected_info_keys or []),
            "previous_best_reward_source": (
                (best_overall or {}).get("source")
            ),
            "reward_reflection": reflection,
        }, indent=2)

        gen_results: list[dict[str, Any]] = []
        for c in range(k):
            cand_dir = gen_dir / f"cand_{c}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            cand_record: dict[str, Any] = {"candidate": c, "fitness": 0.0,
                                           "valid": False, "trained": False,
                                           "attempts": []}
            source = ""
            # One resample on an invalid candidate, then honest zero.
            # Per-attempt records keep the audit trail truthful (an
            # invalid-then-valid candidate must not carry a stale
            # error next to valid=true).
            for attempt in range(2):
                candidates_sampled += 1
                try:
                    source = _sample_candidate(
                        client, system_prompt, user_content, model=model,
                    )
                except Exception as api_err:  # noqa: BLE001 — H2: an API
                    # failure is an invalid candidate, never a dead job
                    # (a crashed job would zero the GPU accounting of
                    # generations that DID train).
                    cand_record["attempts"].append(
                        {"attempt": attempt,
                         "api_error": f"{type(api_err).__name__}: {api_err}"}
                    )
                    continue
                reward_path = cand_dir / "reward.py"
                reward_path.write_text(source, encoding="utf-8")
                static_err = _statically_valid(source)
                if static_err is not None:
                    cand_record["attempts"].append(
                        {"attempt": attempt, "static_error": static_err})
                    continue
                probe = adapter.probe_component(reward_path)
                if probe.ok:
                    cand_record["valid"] = True
                    cand_record["attempts"].append({"attempt": attempt,
                                                    "ok": True})
                    break
                cand_record["attempts"].append(
                    {"attempt": attempt,
                     "probe_error": str(probe.error)[:400]})
            if not cand_record["valid"]:
                cand_record["source"] = source
                gen_results.append(cand_record)
                continue

            iter_dir = cand_dir / "train"
            if iter_dir.exists():
                # A previous crashed harness run may have left partial
                # artifacts (e.g. a dead attempt's reward_trajectory) —
                # a fresh train must not inherit them.
                shutil.rmtree(iter_dir)
            iter_dir.mkdir()
            try:
                train_res = adapter.train(
                    reward_module_path=cand_dir / "reward.py",
                    output_dir=iter_dir,
                    steps=steps_per_iter,
                    seed=seed,
                )
                candidates_trained += 1
                cand_record["trained"] = True
                adapter.rollout(
                    checkpoint_path=train_res.checkpoint_path,
                    output_dir=iter_dir / "rollout",
                    n_episodes=rollout_episodes,
                )
                spec = compute_spec_metrics(spec_metric, iter_dir / "rollout")
                cand_record["fitness"] = float(spec.get("spec_score", 0.0))
                cand_record["spec"] = spec
                cand_record["source"] = source
                cand_record["iter_dir"] = str(iter_dir)
            except Exception as e:  # noqa: BLE001 — honest zero
                cand_record["error"] = f"{type(e).__name__}: {e}"
                cand_record["source"] = source
            gen_results.append(cand_record)

        gen_best = max(gen_results, key=lambda r: r["fitness"], default=None)
        log["generations"].append({
            "generation": g,
            "candidates": [
                {kk: vv for kk, vv in r.items() if kk != "source"}
                for r in gen_results
            ],
            "best_candidate": gen_best["candidate"] if gen_best else None,
            "best_fitness": gen_best["fitness"] if gen_best else 0.0,
        })
        # Flush the audit trail after EVERY generation — a later crash
        # must not erase the record of GPU work already done (H2).
        log["candidates_sampled"] = candidates_sampled
        log["candidates_trained"] = candidates_trained
        from sculptor.run_context import write_json_atomic as _wja

        _wja(job_dir / "eureka_log.json", log)

        # Mirror the generation's best into runs/iter_<g>/ for the
        # harness's standard spec-series machinery.
        mirror = job_dir / "runs" / f"iter_{g}"
        if gen_best is not None and gen_best.get("iter_dir"):
            src = Path(gen_best["iter_dir"])
            if mirror.exists():
                shutil.rmtree(mirror)
            shutil.copytree(src, mirror)
            (mirror / "eureka_candidate.json").write_text(json.dumps({
                "generation": g,
                "candidate": gen_best["candidate"],
                "fitness": gen_best["fitness"],
            }), encoding="utf-8")
        # Eureka keeps the best ACROSS generations as the running
        # context (monotone reflection target).
        if gen_best is not None and gen_best.get("valid") and (
            best_overall is None
            or gen_best["fitness"] > best_overall["fitness"]
        ):
            best_overall = gen_best
            best_overall_iter = (
                Path(gen_best["iter_dir"]) if gen_best.get("iter_dir") else None
            )

    log["candidates_sampled"] = candidates_sampled
    log["candidates_trained"] = candidates_trained
    log["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    from sculptor.run_context import write_json_atomic

    write_json_atomic(job_dir / "eureka_log.json", log)
    return {
        "generations": generations,
        "candidates_sampled": candidates_sampled,
        "candidates_trained": candidates_trained,
    }
