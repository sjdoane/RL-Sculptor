"""Background runners for KG ingest + extract jobs.

Each runner returns an async callable matching `JobManager.submit`'s
`fn` signature. The actual sculptor calls are sync (urllib for arxiv,
LLM for extract), so we wrap them with `asyncio.to_thread`.

Ingest jobs honor cancellation between the two phases (ingest → extract);
we don't interrupt mid-phase because the underlying sculptor functions
aren't cancellation-safe.

Job kinds:
  - `kg_ingest_extract` — seeded from `kg_seeds.yml` (bulk) or a
    library-scaffold auto-seed. Ingest + optional Claude extract.
  - `kg_research` — NEW in M7 P2. Claude picks arxiv IDs for a topic,
    dedupes against the shared KG, then ingests + extracts the new
    ones. Writes to the shared KG regardless of which project
    triggered it (projects that still have a legacy per-project DB
    keep using it; the research endpoint is the one universal write
    path to the shared graph).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

# Eagerly import `sculptor` at module load so its `__init__.py`
# `.env` loader runs before we later check `os.environ.get
# ("ANTHROPIC_API_KEY")`. Pre-fix: kg_jobs.py imported sculptor only
# lazily inside `_do_extract`, so the first backend-side extract
# decision ran BEFORE .env had been loaded, and Sam saw
# "Paper could not be extracted because no API key is configured"
# despite the key being present in ~/projects/RewardSculptor/.env
# (Test 1 round 3 2026-04-22). The import is side-effect-only; we
# don't reference the binding.
import sculptor  # noqa: F401

from backend.services import kg_store
from backend.services.job_manager import Job


def run_ingest_extract_job(
    *,
    project_dir: Path,
    auto_extract: bool,
) -> Callable[[Job, asyncio.Event], Awaitable[dict[str, Any]]]:
    """Return a job callable that:
      1. calls sculptor.kg.ingest.ingest_from_seeds on the project's
         kg_seeds.yml;
      2. if auto_extract is True and the ingest fetched at least one
         paper, calls sculptor.kg.extract.extract_all.
    """

    async def _runner(job: Job, cancel: asyncio.Event) -> dict[str, Any]:
        seeds_path = project_dir / "kg_seeds.yml"
        db_path = kg_store.project_kg_db_path(project_dir)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        from sculptor.kg.ingest import ingest_from_seeds
        from sculptor.kg.store import SculptorKG

        job.progress = 0.05
        job.message = "ingesting arxiv metadata + PDFs"

        def _do_ingest() -> dict[str, Any]:
            with SculptorKG(db_path) as store:
                return ingest_from_seeds(seeds_path, store=store)

        ingest_results = await asyncio.to_thread(_do_ingest)
        job.progress = 0.55
        ingested_count = sum(
            1 for v in ingest_results.values() if v == "ingested"
        )
        already = sum(
            1 for v in ingest_results.values() if v == "already_present"
        )
        errors = {
            aid: reason
            for aid, reason in ingest_results.items()
            if isinstance(reason, str) and reason.startswith("error")
        }
        job.message = (
            f"ingested {ingested_count} new, {already} already present, "
            f"{len(errors)} error(s)"
        )

        extract_summary: dict[str, Any] | None = None
        if auto_extract and not cancel.is_set():
            # Only run extract if the API key is present; otherwise
            # flag it in the message + result rather than erroring.
            import os

            if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
                job.message += " · skipped extract (no ANTHROPIC_API_KEY)"
                extract_summary = {"skipped": True, "reason": "no_api_key"}
            else:
                job.progress = 0.6
                job.message = "extracting entities via Claude"

                def _on_paper_progress(done: int, total: int, title: str) -> None:
                    # Truncate long titles so the job.message fits the
                    # UI chip. The `extracting N / M` prefix is stable
                    # so the frontend can parse it cheaply.
                    short = (title or "").strip()
                    if len(short) > 60:
                        short = short[:57] + "…"
                    job.message = f"extracting {done} / {total}: {short}"
                    # Map paper progress into the 0.6-0.95 range so the
                    # bar keeps moving even when extract dominates wall
                    # time.
                    if total > 0:
                        job.progress = 0.6 + 0.35 * (done / total)

                def _do_extract() -> dict[str, Any]:
                    from sculptor.kg.extract import extract_all

                    with SculptorKG(db_path) as store:
                        results = extract_all(
                            store=store, progress_cb=_on_paper_progress,
                        )
                    summary = {
                        "attempted": len(results),
                        "succeeded": sum(1 for r in results if not r.error and not r.skipped),
                        "skipped":  sum(1 for r in results if r.skipped),
                        "errored":  sum(1 for r in results if r.error),
                        "errors":   [
                            {"paper_id": r.paper_id, "error": r.error}
                            for r in results if r.error
                        ][:10],
                    }
                    return summary

                extract_summary = await asyncio.to_thread(_do_extract)
                job.message = (
                    f"extracted {extract_summary.get('succeeded', 0)} paper(s)"
                )

        job.progress = 1.0
        techniques_count = 0
        try:
            stats = kg_store.get_stats(project_dir)
            techniques_count = stats.techniques
        except Exception:  # noqa: BLE001
            pass

        return {
            "ingest": {
                "ingested": ingested_count,
                "already_present": already,
                "errors": errors,
                "results": ingest_results,
            },
            "extract": extract_summary,
            "kg_techniques_count": techniques_count,
        }

    return _runner


def run_research_job(
    *,
    project_dir: Path,
    topic: str,
    max_papers: int = 10,
    auto_extract: bool = True,
) -> Callable[[Job, asyncio.Event], Awaitable[dict[str, Any]]]:
    """Return a job callable that:
      1. Calls `sculptor.kg.research.research_topic(topic)` → Claude
         returns a deduped list of new arxiv IDs relevant to the topic.
      2. For each new ID, calls `sculptor.kg.ingest.ingest_arxiv` into
         the shared KG (or legacy per-project DB if one still exists).
      3. Optionally fires `sculptor.kg.extract.extract_all` on the
         shared KG so new papers ship with entities on day one.

    `project_dir` is only used for KG path resolution — the research
    output targets the shared KG (or the legacy per-project DB if that
    project still has one), mirroring how the rest of the KG routes
    resolve paths.
    """

    async def _runner(job: Job, cancel: asyncio.Event) -> dict[str, Any]:
        # ── Stage 1: Claude research call ─────────────────────────────
        job.progress = 0.05
        job.message = f"asking Claude for papers on: {topic!r}"

        def _do_research():
            from sculptor.kg.research import research_topic
            from sculptor.kg.store import SculptorKG

            db_path = kg_store.project_kg_db_path(project_dir)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with SculptorKG(db_path) as store:
                return research_topic(
                    topic, max_papers=max_papers, store=store,
                    dedupe_against_kg=True,
                )

        research_result = await asyncio.to_thread(_do_research)
        new_ids = [p.arxiv_id for p in research_result.papers]
        job.params["research_ids"] = new_ids
        job.params["research_coverage_note"] = research_result.coverage_note
        # Issue D (Test 1 2026-04-22) — stage counters so the UI can
        # distinguish "Claude returned 0" from "Claude returned 5, all
        # already in KG" instead of showing a vague "added 0" check.
        job.params["papers_returned_by_claude"] = research_result.papers_returned_by_claude
        job.params["papers_deduped_against_kg"] = research_result.papers_deduped_against_kg

        if not new_ids:
            job.progress = 1.0
            # Targeted message per the stage that ate the papers.
            claude_n = research_result.papers_returned_by_claude
            dedup_n = research_result.papers_deduped_against_kg
            if claude_n == 0:
                job.message = (
                    f"Claude returned 0 papers for {topic!r} — try a "
                    f"more specific query or check coverage_note"
                )
            elif dedup_n > 0:
                job.message = (
                    f"Claude returned {claude_n} paper(s), all {dedup_n} "
                    f"already in KG (0 new to ingest)"
                )
            else:
                job.message = (
                    "no new papers — topic is already covered in the KG"
                )
            return {
                "topic": topic,
                "new_papers": [],
                "coverage_note": research_result.coverage_note,
                "papers_returned_by_claude": claude_n,
                "papers_deduped_against_kg": dedup_n,
                "ingest": None,
                "extract": None,
            }

        job.progress = 0.2
        job.message = (
            f"found {len(new_ids)} new paper(s); ingesting PDFs"
        )

        # ── Stage 2: ingest the new IDs ───────────────────────────────
        def _do_ingest():
            from sculptor.kg.ingest import ingest_arxiv
            from sculptor.kg.store import SculptorKG

            db_path = kg_store.project_kg_db_path(project_dir)
            ingested: dict[str, str] = {}
            with SculptorKG(db_path) as store:
                for aid in new_ids:
                    try:
                        ingest_arxiv(aid, store=store, force=False)
                        ingested[aid] = "ingested"
                    except Exception as e:  # noqa: BLE001
                        ingested[aid] = f"error: {type(e).__name__}: {e}"
            return ingested

        ingest_results = await asyncio.to_thread(_do_ingest)
        ingested_ok = [
            aid for aid, v in ingest_results.items() if v == "ingested"
        ]
        ingest_errors = {
            aid: v for aid, v in ingest_results.items()
            if isinstance(v, str) and v.startswith("error")
        }

        extract_summary: dict[str, Any] | None = None
        if auto_extract and ingested_ok and not cancel.is_set():
            import os

            if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
                job.message = (
                    f"ingested {len(ingested_ok)} paper(s); skipped "
                    "extract (no ANTHROPIC_API_KEY)"
                )
                extract_summary = {"skipped": True, "reason": "no_api_key"}
            else:
                job.progress = 0.7
                job.message = (
                    f"extracting entities for {len(ingested_ok)} new paper(s)"
                )

                def _do_extract():
                    from sculptor.kg.extract import extract_all
                    from sculptor.kg.store import SculptorKG

                    db_path = kg_store.project_kg_db_path(project_dir)
                    with SculptorKG(db_path) as store:
                        results = extract_all(store=store)
                    return {
                        "attempted": len(results),
                        "succeeded": sum(
                            1 for r in results if not r.error and not r.skipped
                        ),
                        "skipped": sum(1 for r in results if r.skipped),
                        "errored": sum(1 for r in results if r.error),
                    }

                extract_summary = await asyncio.to_thread(_do_extract)

        job.progress = 1.0
        job.message = (
            f"research complete: +{len(ingested_ok)} paper(s) on {topic!r}"
        )
        return {
            "topic": topic,
            "new_papers": [
                p.model_dump() for p in research_result.papers
            ],
            "coverage_note": research_result.coverage_note,
            "ingest": {
                "ingested": ingested_ok,
                "errors": ingest_errors,
            },
            "extract": extract_summary,
        }

    return _runner
