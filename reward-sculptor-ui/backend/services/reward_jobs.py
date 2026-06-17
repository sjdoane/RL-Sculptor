"""Background runners for reward-related jobs.

Currently hosts the `reward_prompt_edit` job (M7 P3): user types a
natural-language prompt on the Rewards tab, Claude rewrites v<n>.py
without a full sculpt-run training loop. Mirrors `kg_jobs.py`'s
`asyncio.to_thread` pattern because the underlying sculptor call
is synchronous.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from backend.services.job_manager import Job


# Wall-clock ceiling for a single prompt-edit job (seconds). Wraps the
# sync `apply_prompt_edit` inside `asyncio.wait_for` so Sam's UI never
# hangs forever on a wedged Anthropic call.
#
# Budget math (post 2026-04-23 sizing):
#   - Claude Opus with adaptive-thinking + 16K max_tokens on a ~7K-char
#     reward-edit prompt takes 180-240s per call (observed 204s live).
#   - `apply_edits` does up to TWO attempts: attempt 1, post-validate,
#     retry once on validation error (SyntaxError, schema mismatch, etc).
#   - Anthropic SDK itself retries twice on transient 429/500 inside
#     each call, adding up to ~60s of backoff.
#   Worst case: 2 × (240 + 60) = 600s. Add 5 min margin → 900s.
# Pre-2026-04-23 default was 300s which was too tight — Sam hit the
# exact failure mode at 12:57: attempt 1 completed in 204s, attempt 2
# had only ~90s left and timed out mid-LLM-call.
# Overridable via env for CI (where we want fast-fail) and for long-
# prompt / experimental runs.
DEFAULT_PROMPT_EDIT_TIMEOUT_S = float(
    os.environ.get("RS_REWARD_PROMPT_TIMEOUT_S", "900")
)


def run_reward_prompt_edit_job(
    *,
    project_dir: Path,
    user_prompt: str,
    expected_parent_version: int,
    timeout_s: float | None = None,
) -> Callable[[Job, asyncio.Event], Awaitable[dict[str, Any]]]:
    """Return a job callable that:
      1. Resolves the latest `rewards/v<N>.py` and asserts it matches
         `expected_parent_version` (optimistic concurrency like the
         manual PUT edit endpoint).
      2. Loads the project's adapter to get the reward_contract.
      3. Calls `sculptor.edit.apply_prompt_edit` → Claude writes
         `rewards/v<N+1>.py` + rewrites `current.py`.
      4. Returns `{new_version, new_path}`.

    The job is marked errored on any EditValidationError, ConcurrencyError,
    or bare LLM failure — caller polls `/jobs/{job_id}` for the outcome.
    """

    budget_s = timeout_s if timeout_s is not None else DEFAULT_PROMPT_EDIT_TIMEOUT_S

    async def _runner(job: Job, cancel: asyncio.Event) -> dict[str, Any]:
        job.progress = 0.05
        job.message = "validating parent + loading adapter"
        job.emit({
            "type": "log_line",
            "text": "[reward_prompt_edit] start — validating parent + loading adapter",
        })

        # Thread-safe bridge: sculptor's `on_event` fires from the
        # to_thread worker; marshal back to the job's loop so `emit`
        # (which touches subscribers' asyncio.Queue) is called on the
        # right thread. Falls back to in-thread emit if no loop bound
        # (tests that call _runner directly without JobManager).
        loop = asyncio.get_running_loop()

        def _emit_from_worker(ev: dict[str, Any]) -> None:
            try:
                loop.call_soon_threadsafe(job.emit, ev)
            except RuntimeError:
                # Loop already closed — best effort.
                job.emit(ev)

        def _do_edit() -> dict[str, Any]:
            # §Ship-10 — granular progress events so the UI has
            # something to show between "dispatching to Claude" and
            # the first event `apply_prompt_edit` emits. Pre-fix, a
            # cold sentence-transformers load could silently hang for
            # 60-120s on WSL2; users saw no progress and assumed
            # Claude was wedged. Each step below emits one breadcrumb.
            _emit_from_worker({
                "type": "log_line",
                "text": "[reward_prompt_edit] loading adapter + reward_contract",
            })
            from sculptor.adapters.base import load_adapter
            from sculptor.edit import apply_prompt_edit
            from sculptor.kg.store import SculptorKG

            # Resolve latest v<n>.py in rewards/.
            rewards_dir = project_dir / "rewards"
            if not rewards_dir.is_dir():
                raise RuntimeError(
                    f"rewards/ not found at {rewards_dir} — "
                    f"project must be scaffolded first."
                )
            latest_n, _latest_path = _find_latest(rewards_dir)
            if latest_n != expected_parent_version:
                raise _ConcurrencyError(
                    expected=expected_parent_version, current=latest_n
                )

            # Load the adapter so we have a reward_contract.
            config_path = project_dir / "config.toml"
            adapter = load_adapter(config_path)
            reward_contract = adapter.reward_contract()

            # Open the shared-or-legacy KG for citation validation.
            from backend.services import kg_store as kg_store_svc
            db_path = kg_store_svc.project_kg_db_path(project_dir)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _emit_from_worker({
                "type": "log_line",
                "text": f"[reward_prompt_edit] opened KG at {db_path.name}",
            })

            with SculptorKG(db_path) as store:
                # Pre-query the KG with the same parameters apply_prompt_edit
                # uses internally, so we can surface the matches on the
                # result dict. apply_prompt_edit queries again with its
                # own min_similarity floor; results are deterministic
                # given the same KG + query so this is effectively free
                # on a warm embedding cache. Visibility for Sam's
                # "nowhere shows what the KG was consulted for" concern
                # (Test 1 round 4, 2026-04-22).
                from sculptor.kg.query import query_semantic
                _MIN_SIM = 0.35  # must match sculptor.edit._MIN_PROMPT_EDIT_SIMILARITY
                _emit_from_worker({
                    "type": "log_line",
                    "text": (
                        "[reward_prompt_edit] KG preview query "
                        "(first call may take 60-120s on cold embedding model)"
                    ),
                })
                import time as _time
                _t0 = _time.time()
                try:
                    kg_preview_matches = query_semantic(
                        user_prompt, top_k=5, store=store,
                        min_similarity=_MIN_SIM,
                    )
                except Exception as e:  # noqa: BLE001
                    _emit_from_worker({
                        "type": "log_line",
                        "text": (
                            f"[reward_prompt_edit] KG preview failed "
                            f"({type(e).__name__}: {e}) — continuing "
                            f"without preview"
                        ),
                    })
                    kg_preview_matches = []
                _emit_from_worker({
                    "type": "log_line",
                    "text": (
                        f"[reward_prompt_edit] KG preview done "
                        f"({len(kg_preview_matches)} matches ≥ sim={_MIN_SIM:.2f}, "
                        f"{_time.time() - _t0:.1f}s)"
                    ),
                })

                _emit_from_worker({
                    "type": "log_line",
                    "text": (
                        "[reward_prompt_edit] delegating to sculptor.edit.apply_prompt_edit "
                        "(KG re-query + Claude LLM call + validate)"
                    ),
                })
                new_path = apply_prompt_edit(
                    current_reward_path=rewards_dir / f"v{latest_n}.py",
                    user_prompt=user_prompt,
                    new_iter_id=f"v{latest_n + 1}",
                    reward_contract=reward_contract,
                    kg_store=store,
                    on_event=_emit_from_worker,
                )

            # Flatten lit_context into a UI-renderable list. Only the
            # fields the frontend needs: technique name, relevance, the
            # arxiv_ids that introduced it (from `paper:<id>` form),
            # and the paper_citation for hover-display. Empty list when
            # no matches crossed the similarity floor — UI can then
            # show "no KG matches for this prompt" explicitly.
            kg_match_payload = [
                {
                    "technique": m.technique.name,
                    "relevance_score": round(float(m.relevance_score), 3),
                    "arxiv_ids": [
                        pid[len("paper:"):] for pid in (m.source_paper_ids or [])
                        if pid.startswith("paper:")
                    ],
                    "paper_citation": m.paper_citation,
                }
                for m in kg_preview_matches
            ]
            return {
                "new_version": latest_n + 1,
                "new_path": str(new_path),
                "parent_version": latest_n,
                "kg_matches": kg_match_payload,
                "kg_min_similarity": _MIN_SIM,
            }

        job.progress = 0.2
        job.message = "Claude is rewriting the reward"
        job.emit({
            "type": "log_line",
            "text": f"[reward_prompt_edit] dispatching to Claude (timeout={budget_s:.0f}s)",
        })

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_do_edit), timeout=budget_s,
            )
        except asyncio.TimeoutError as e:
            job.emit({
                "type": "log_line",
                "text": f"[reward_prompt_edit] timeout after {budget_s:.0f}s — aborting",
            })
            raise RuntimeError(
                f"reward-prompt edit timed out after {budget_s:.0f}s "
                f"(RS_REWARD_PROMPT_TIMEOUT_S). Claude call is likely "
                f"wedged; the lock is now released."
            ) from e
        except _ConcurrencyError as e:
            raise RuntimeError(
                f"parent version changed under us: expected "
                f"v{e.expected}, latest is v{e.current}. "
                f"Refresh the Rewards tab and retry."
            ) from e

        job.progress = 1.0
        job.message = (
            f"wrote v{result['new_version']}.py from prompt"
        )
        job.emit({
            "type": "log_line",
            "text": f"[reward_prompt_edit] completed v{result['new_version']}.py",
        })
        return result

    return _runner


# ── helpers ───────────────────────────────────────────────────────────
class _ConcurrencyError(Exception):
    def __init__(self, *, expected: int, current: int):
        super().__init__(
            f"expected parent v{expected}, but latest on disk is v{current}"
        )
        self.expected = expected
        self.current = current


def _find_latest(rewards_dir: Path) -> tuple[int, Path]:
    """Return (N, path_to_vN.py) for the highest-numbered version file."""
    best_n = -1
    best_path: Path | None = None
    for p in rewards_dir.glob("v*.py"):
        m = re.fullmatch(r"v(\d+)", p.stem)
        if m is None:
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n = n
            best_path = p
    if best_path is None:
        raise RuntimeError(
            f"no v<n>.py reward file in {rewards_dir}; "
            "scaffold the project via `sculpt init` or the UI."
        )
    return best_n, best_path


def sha256_prefix(text: str, *, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


# ── physics prompt edit (Phase 5) ─────────────────────────────────────
def run_physics_prompt_edit_job(
    *,
    project_dir: Path,
    user_prompt: str,
) -> Callable[[Job, asyncio.Event], Awaitable[dict[str, Any]]]:
    """Return a job callable that dispatches the Claude-backed MJCF
    rewrite. The underlying call in `backend.services.physics.
    apply_prompt_edit` handles materialization, Claude, validation,
    write, and git commit. Returns a summary dict the UI can render
    (new XML, diff, commit sha, rejected_reason)."""

    async def _runner(job: Job, cancel: asyncio.Event) -> dict[str, Any]:
        job.progress = 0.1
        job.message = "loading MJCF + asking Claude to rewrite"

        def _do_edit() -> dict[str, Any]:
            from backend.services import kg_store as kg_store_svc
            from backend.services.physics import apply_prompt_edit
            from sculptor.kg.store import SculptorKG

            # Open the project's KG (shared-first path) so Claude's
            # physics rewrite is grounded in literature — mandated by
            # the "every change consults the KG" policy (QUALITY_PASS_PLAN
            # Phase B). `apply_prompt_edit` degrades gracefully when
            # `kg_store=None` or the KG query throws, so any failure
            # here is never fatal.
            db_path = kg_store_svc.project_kg_db_path(project_dir)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with SculptorKG(db_path) as kg:
                return apply_prompt_edit(
                    project_dir, user_prompt, kg_store=kg,
                )

        result = await asyncio.to_thread(_do_edit)
        job.progress = 1.0
        if result.get("committed"):
            job.message = f"physics: {result.get('summary', 'edit committed')}"
            # Invalidate the cached preview PNG so the Overview tab's
            # static render reflects the new MJCF on next refresh.
            # Pre-fix (Test 1 round 3 2026-04-22): Sam committed a
            # physics edit adding a "hanging pole" keyframe but the
            # Overview preview still showed the default pose because
            # robot.py's preview route only checks file existence, not
            # mtime / MJCF hash. Inline the glob-unlink (equivalent to
            # ProjectStore.invalidate_preview but the runner has
            # project_dir directly, not the slug-indexed store singleton).
            uploads = project_dir / "uploads"
            if uploads.is_dir():
                for p in uploads.glob("preview_*.png"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
        else:
            reason = result.get("rejected_reason") or "no change"
            job.message = f"rejected — {reason}"
        return result

    return _runner
