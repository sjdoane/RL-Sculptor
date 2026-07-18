"""sculptor/kg/extract.py — LLM-powered entity extraction.

For each Paper in the KG, ask Claude (the registry's "kg_extract" role,
`sculptor.llm.model_for`) to extract:
  * Techniques (named methods or design patterns)
  * Failure modes (training pathologies the paper discusses)
  * Reward components (reusable reward-shaping terms)
  * Environments (benchmark envs / robot platforms)

…plus the four relations Sculptor's diagnoser needs:
  * Paper  INTRODUCES   Technique
  * Technique ADDRESSES FailureMode
  * Technique USES      RewardComponent
  * Paper  EVALUATES_ON Environment

Every entity carries a verbatim evidence snippet from the source text. Strict
JSON via `client.messages.parse()` + Pydantic. One retry on parse failure
with a corrective reminder.

Idempotent: skips Papers with `extracted=True` unless `--force` is passed.

CLI:
    uv run sculpt kg extract --all
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from sculptor.kg.schema import (
    Edge,
    Environment,
    FailureMode,
    Paper,
    PROVENANCE_PAPER_CLAIM,
    Relation,
    RewardComponent,
    Technique,
    make_environment_id,
    make_failure_mode_id,
    make_paper_id,
    make_reward_component_id,
    make_technique_id,
    merge_provenance,
)
from sculptor.kg.store import SculptorKG
from sculptor.llm import log_llm_call, model_for, response_text_blocks


MODEL_ID = model_for("kg_extract")
MAX_TOKENS = 8192
MAX_EXCERPT_CHARS = 28_000    # ~7K tokens — enough for most papers' key sections
MIN_EVIDENCE_CHARS = 20       # reject snippets that are just category names
RETRY_REMINDER = (
    "Your previous response did not validate against the schema. "
    "Return the extraction JSON only — no preamble, no markdown code fence."
)


# ── Pydantic schema Claude must honor ──────────────────────────────────────
class TechniqueExtract(BaseModel):
    name: str = Field(description="short snake_case slug, e.g. reference_state_initialization")
    description: str
    tags: list[str] = Field(default_factory=list)
    evidence: str = Field(description="1–2 sentence verbatim snippet from the paper")


class FailureModeExtract(BaseModel):
    name: str = Field(description="short snake_case slug, e.g. walking_not_jumping")
    description: str
    symptoms: list[str] = Field(default_factory=list)
    environment_tag: Optional[str] = None
    evidence: str


class RewardComponentExtract(BaseModel):
    name: str
    description: str
    formula: Optional[str] = None
    hyperparameters: dict[str, float] = Field(default_factory=dict)
    evidence: str


class EnvironmentExtract(BaseModel):
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    evidence: str


class PaperToTechniqueRel(BaseModel):
    technique: str


class TechniqueToFailureModeRel(BaseModel):
    technique: str
    failure_mode: str


class TechniqueToRewardComponentRel(BaseModel):
    technique: str
    reward_component: str


class PaperToEnvironmentRel(BaseModel):
    environment: str


class ExtractionPayload(BaseModel):
    """Top-level structure Claude returns for a single paper."""

    techniques: list[TechniqueExtract] = Field(default_factory=list)
    failure_modes: list[FailureModeExtract] = Field(default_factory=list)
    reward_components: list[RewardComponentExtract] = Field(default_factory=list)
    environments: list[EnvironmentExtract] = Field(default_factory=list)
    paper_to_technique: list[PaperToTechniqueRel] = Field(default_factory=list)
    technique_to_failure_mode: list[TechniqueToFailureModeRel] = Field(default_factory=list)
    technique_to_reward_component: list[TechniqueToRewardComponentRel] = Field(default_factory=list)
    paper_to_environment: list[PaperToEnvironmentRel] = Field(default_factory=list)


# ── Plain-python result for programmatic callers ─────────────────────────────
@dataclass
class ExtractionResult:
    paper_id: str
    skipped: bool = False
    error: str | None = None
    node_ids: list[str] = field(default_factory=list)
    edge_count: int = 0
    raw_payload: dict | None = None  # decoded Pydantic dict, for inspection/debug
    elapsed_s: float = 0.0


# ── System prompt ────────────────────────────────────────────────────────────
from sculptor.prompts import load_prompt

_SYSTEM_PROMPT = load_prompt("kg_extract")


# ── Input assembly ──────────────────────────────────────────────────────────
def _build_excerpt(paper: Paper) -> str:
    """Abstract + conclusion + a bounded slice of full text.

    Full text (when available) is included because the abstract/conclusion
    alone often omit named failure modes and specific reward components.
    """
    parts: list[str] = []
    parts.append(f"# TITLE\n{paper.title}")
    if paper.authors:
        parts.append(f"# AUTHORS\n{', '.join(paper.authors)}")
    if paper.abstract:
        parts.append(f"# ABSTRACT\n{paper.abstract}")
    if paper.conclusion_text:
        parts.append(f"# CONCLUSION\n{paper.conclusion_text}")

    # Dip into the full-text sidecar for a bounded middle slice. We skip the
    # first ~1K chars (often residual header/metadata) and cap the excerpt so
    # the whole prompt stays well under the context window.
    if paper.full_text_path and os.path.isfile(paper.full_text_path):
        try:
            full = Path(paper.full_text_path).read_text(
                encoding="utf-8", errors="replace")
            budget = max(0, MAX_EXCERPT_CHARS - sum(len(p) for p in parts))
            if budget > 500:
                body = full[1000: 1000 + budget]
                parts.append(f"# BODY_EXCERPT\n{body}")
        except Exception:
            pass

    excerpt = "\n\n".join(parts)
    if len(excerpt) > MAX_EXCERPT_CHARS:
        excerpt = excerpt[:MAX_EXCERPT_CHARS]
    return excerpt


# ── LLM call ────────────────────────────────────────────────────────────────
def _call_claude(client, system_prompt: str, user_prompt: str) -> ExtractionPayload:
    """One LLM call with strict-JSON parsing; one retry on parse/validation failure."""
    # First try: standard request.
    try:
        resp = client.messages.parse(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            cache_control={"type": "ephemeral"},  # auto-caches the last cacheable block
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=ExtractionPayload,
        )
        log_llm_call(
            "kg_extract", MODEL_ID, system=system_prompt, user=user_prompt,
            response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None), meta={"attempt": 1})
        return resp.parsed_output  # type: ignore[return-value]
    except Exception as first_err:
        # Retry once with a corrective reminder appended as a follow-up user turn.
        retry_reminder = f"{RETRY_REMINDER} Error was: {first_err!s}"
        corrected_messages = [
            {"role": "user", "content": user_prompt},
            # 4.7 disallows assistant prefill; use a second user turn instead.
            {"role": "user", "content": retry_reminder},
        ]
        # Second attempt: re-run with the nudge; propagate any failure.
        resp = client.messages.parse(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=corrected_messages,
            output_format=ExtractionPayload,
        )
        log_llm_call(
            "kg_extract", MODEL_ID, system=system_prompt,
            user=f"{user_prompt}\n\n{retry_reminder}",
            response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None), meta={"attempt": 2})
        return resp.parsed_output  # type: ignore[return-value]


# ── Materialize extraction into the KG ──────────────────────────────────────
def _materialize(
    store: SculptorKG,
    paper: Paper,
    payload: ExtractionPayload,
) -> tuple[list[str], int]:
    """Upsert nodes + edges; return (node_ids, edge_count)."""
    created_node_ids: list[str] = []

    def _good_evidence(ev: str) -> bool:
        return isinstance(ev, str) and len(ev.strip()) >= MIN_EVIDENCE_CHARS

    # 1. Techniques
    #
    # MERGE DISCIPLINE (§Phase-0 hardening 2026-07-18): when the node
    # already exists, mutate it with dataclasses.replace so every field
    # this payload does NOT speak to survives. The old rebuild-from-
    # scratch merge silently reset `useful_citations` and `outcome_stats`
    # — the run-learned ranking signals cases.record_run_cases spends
    # whole GPU runs accumulating — to 0/{} on every re-extraction that
    # mentioned the same technique. Provenance merges by trust tier
    # (a paper attestation upgrades an llm-inferred stub, never the
    # reverse). Same pattern for FailureMode/RewardComponent/Environment.
    technique_name_to_id: dict[str, str] = {}
    for t in payload.techniques:
        if not _good_evidence(t.evidence):
            continue
        tid = make_technique_id(t.name)
        existing = store.get_node(tid)
        if isinstance(existing, Technique):
            node = dataclasses.replace(
                existing,
                description=(existing.description
                             if len(existing.description) > len(t.description)
                             else t.description),
                tags=sorted(set(existing.tags) | set(t.tags)),
                provenance=merge_provenance(
                    existing.provenance, PROVENANCE_PAPER_CLAIM),
            )
        else:
            node = Technique(
                id=tid, name=t.name, description=t.description,
                tags=sorted(set(t.tags)))
        store.add_node(node)
        technique_name_to_id[t.name] = tid
        created_node_ids.append(tid)

    # 2. Failure modes
    failure_name_to_id: dict[str, str] = {}
    for f in payload.failure_modes:
        if not _good_evidence(f.evidence):
            continue
        fid = make_failure_mode_id(f.name)
        existing = store.get_node(fid)
        if isinstance(existing, FailureMode):
            # A diagnoser-flagged stub carries the placeholder description;
            # a real paper description (even a shorter one) beats it.
            _stub = existing.description.startswith("(diagnoser-flagged")
            node = dataclasses.replace(
                existing,
                description=(f.description if _stub or
                             len(f.description) > len(existing.description)
                             else existing.description),
                symptoms=sorted(set(existing.symptoms) | set(f.symptoms)),
                environment_tag=existing.environment_tag or f.environment_tag,
                provenance=merge_provenance(
                    existing.provenance, PROVENANCE_PAPER_CLAIM),
            )
        else:
            node = FailureMode(
                id=fid, name=f.name, description=f.description,
                symptoms=sorted(set(f.symptoms)),
                environment_tag=f.environment_tag)
        store.add_node(node)
        failure_name_to_id[f.name] = fid
        created_node_ids.append(fid)

    # 3. Reward components
    reward_component_name_to_id: dict[str, str] = {}
    for r in payload.reward_components:
        if not _good_evidence(r.evidence):
            continue
        rid = make_reward_component_id(r.name)
        existing = store.get_node(rid)
        if isinstance(existing, RewardComponent):
            node = dataclasses.replace(
                existing,
                description=(existing.description
                             if len(existing.description) > len(r.description)
                             else r.description),
                formula=existing.formula or r.formula,
                hyperparameters={**existing.hyperparameters,
                                 **r.hyperparameters},
                provenance=merge_provenance(
                    existing.provenance, PROVENANCE_PAPER_CLAIM),
            )
        else:
            node = RewardComponent(
                id=rid, name=r.name, description=r.description,
                formula=r.formula, hyperparameters=dict(r.hyperparameters))
        store.add_node(node)
        reward_component_name_to_id[r.name] = rid
        created_node_ids.append(rid)

    # 4. Environments
    environment_name_to_id: dict[str, str] = {}
    for e in payload.environments:
        if not _good_evidence(e.evidence):
            continue
        eid = make_environment_id(e.name)
        existing = store.get_node(eid)
        if isinstance(existing, Environment):
            node = dataclasses.replace(
                existing,
                description=(existing.description
                             if len(existing.description) > len(e.description)
                             else e.description),
                tags=sorted(set(existing.tags) | set(e.tags)),
            )
        else:
            node = Environment(
                id=eid, name=e.name, description=e.description,
                tags=sorted(set(e.tags)))
        store.add_node(node)
        environment_name_to_id[e.name] = eid
        created_node_ids.append(eid)

    # Helpers: look up a relation target by name, falling back to the store
    # when the target entity was introduced by a different paper's payload.
    def _lookup_technique(name: str) -> str | None:
        if name in technique_name_to_id:
            return technique_name_to_id[name]
        tid = make_technique_id(name)
        return tid if store.get_node(tid) is not None else None

    def _lookup_failure_mode(name: str) -> str | None:
        if name in failure_name_to_id:
            return failure_name_to_id[name]
        fid = make_failure_mode_id(name)
        return fid if store.get_node(fid) is not None else None

    def _lookup_reward_component(name: str) -> str | None:
        if name in reward_component_name_to_id:
            return reward_component_name_to_id[name]
        rid = make_reward_component_id(name)
        return rid if store.get_node(rid) is not None else None

    def _lookup_environment(name: str) -> str | None:
        if name in environment_name_to_id:
            return environment_name_to_id[name]
        eid = make_environment_id(name)
        return eid if store.get_node(eid) is not None else None

    # Edge construction — evidence carried in the edge's data blob.
    edge_count = 0

    def _add_supported_edge(
        src: str, dst: str, relation: Relation, evidence: str,
    ) -> None:
        """Upsert a many-paper claim without erasing corroboration.

        The edge table intentionally has one row per (src, dst, relation),
        so claim-level provenance belongs in a `supports` list inside the
        data blob. Legacy single-source fields remain mirrored for older
        readers, while new readers can keep citation and evidence aligned.
        """
        existing_data: dict[str, Any] = {}
        for old_edge, other in store.neighbors(
                src, relation=relation, direction="out"):
            if other == dst:
                existing_data = dict(old_edge.data or {})
                break
        by_paper: dict[str, str] = {}
        for support in existing_data.get("supports", []) or []:
            if not isinstance(support, dict):
                continue
            pid = str(support.get("source_paper_id") or "")
            if pid:
                by_paper[pid] = str(support.get("evidence") or "")
        legacy_pid = str(existing_data.get("source_paper_id") or "")
        if legacy_pid:
            by_paper.setdefault(
                legacy_pid, str(existing_data.get("evidence") or ""))
        if evidence or paper.id not in by_paper:
            by_paper[paper.id] = evidence
        supports = [
            {"source_paper_id": pid, "evidence": ev}
            for pid, ev in sorted(by_paper.items())
        ]
        store.add_edge(Edge(
            src=src, dst=dst, relation=relation,
            data={
                "evidence": by_paper.get(paper.id, evidence),
                "source_paper_id": paper.id,
                "supports": supports,
            },
        ))

    # Paper INTRODUCES Technique
    technique_evidence = {t.name: t.evidence for t in payload.techniques if _good_evidence(t.evidence)}
    for rel in payload.paper_to_technique:
        tid = _lookup_technique(rel.technique)
        if not tid:
            continue
        store.add_edge(Edge(
            src=paper.id, dst=tid, relation=Relation.INTRODUCES,
            data={"evidence": technique_evidence.get(rel.technique, "")},
        ))
        edge_count += 1

    # Technique ADDRESSES FailureMode
    failure_evidence = {f.name: f.evidence for f in payload.failure_modes if _good_evidence(f.evidence)}
    for rel in payload.technique_to_failure_mode:
        tid = _lookup_technique(rel.technique)
        fid = _lookup_failure_mode(rel.failure_mode)
        if not (tid and fid):
            continue
        _add_supported_edge(
            tid, fid, Relation.ADDRESSES,
            failure_evidence.get(rel.failure_mode, ""))
        edge_count += 1

    # Technique USES RewardComponent
    rc_evidence = {r.name: r.evidence for r in payload.reward_components if _good_evidence(r.evidence)}
    for rel in payload.technique_to_reward_component:
        tid = _lookup_technique(rel.technique)
        rid = _lookup_reward_component(rel.reward_component)
        if not (tid and rid):
            continue
        _add_supported_edge(
            tid, rid, Relation.USES,
            rc_evidence.get(rel.reward_component, ""))
        edge_count += 1

    # Paper EVALUATES_ON Environment
    env_evidence = {e.name: e.evidence for e in payload.environments if _good_evidence(e.evidence)}
    for rel in payload.paper_to_environment:
        eid = _lookup_environment(rel.environment)
        if not eid:
            continue
        store.add_edge(Edge(
            src=paper.id, dst=eid, relation=Relation.EVALUATES_ON,
            data={"evidence": env_evidence.get(rel.environment, "")},
        ))
        edge_count += 1

    return created_node_ids, edge_count


# ── Public API ──────────────────────────────────────────────────────────────
def extract_entities(
    paper: Paper,
    *,
    store: SculptorKG | None = None,
    client=None,
    force: bool = False,
) -> ExtractionResult:
    """Extract KG entities + edges from one Paper. Idempotent on `paper.extracted`."""
    if paper.extracted and not force:
        return ExtractionResult(paper_id=paper.id, skipped=True)

    owns_store = store is None
    store = store or SculptorKG()
    t0 = time.time()
    try:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()

        excerpt = _build_excerpt(paper)
        user_prompt = (
            f"<paper>\n"
            f"<arxiv_id>{paper.arxiv_id}</arxiv_id>\n"
            f"<text>\n{excerpt}\n</text>\n"
            f"</paper>\n"
            f"Extract per the instructions."
        )
        payload = _call_claude(client, _SYSTEM_PROMPT, user_prompt)

        node_ids, edge_count = _materialize(store, paper, payload)

        # Mark the paper extracted
        paper.extracted = True
        store.add_node(paper, upsert=True)

        return ExtractionResult(
            paper_id=paper.id,
            node_ids=node_ids,
            edge_count=edge_count,
            raw_payload=payload.model_dump(),
            elapsed_s=time.time() - t0,
        )
    except Exception as e:  # noqa: BLE001
        return ExtractionResult(
            paper_id=paper.id, error=f"{type(e).__name__}: {e}",
            elapsed_s=time.time() - t0)
    finally:
        if owns_store:
            store.close()


def extract_all(
    *,
    store: SculptorKG | None = None,
    force: bool = False,
    limit: int | None = None,
    paper_ids: set[str] | None = None,
    tier: str | None = None,
    tags: set[str] | None = None,
    progress_cb: "Callable[[int, int, str], None] | None" = None,
) -> list[ExtractionResult]:
    """Run extraction over every Paper in the store.

    Returns per-paper results. `limit` caps the number extracted (useful for
    cheap first runs).

    `progress_cb(done, total, title)` is invoked BEFORE each paper's
    extraction starts, where `done` is the 1-indexed count of the
    upcoming paper and `total` is the length of the work queue (after
    filtering for skipped papers). Callback failures are swallowed —
    progress reporting never aborts extraction.
    """
    owns_store = store is None
    store = store or SculptorKG()
    results: list[ExtractionResult] = []
    try:
        papers: list[Paper] = store.find_nodes(kind=Paper.kind)  # type: ignore[assignment]
        if paper_ids is not None:
            present_ids = {p.id for p in papers}
            for missing_id in sorted(paper_ids - present_ids):
                results.append(ExtractionResult(
                    paper_id=missing_id,
                    error="selected campaign Paper is absent from this store"))
            papers = [p for p in papers if p.id in paper_ids]
        if tier:
            papers = [p for p in papers
                      if (p.tier or "").upper() == tier.upper()]
        if tags:
            wanted = {t.strip().lower() for t in tags if t.strip()}
            papers = [p for p in papers
                      if wanted.issubset({t.lower() for t in p.tags})]
        # Extract unextracted first, then (if force) re-extract already-extracted.
        papers.sort(key=lambda p: (p.extracted, p.arxiv_id))

        # Pre-count papers that will actually be processed so the progress
        # callback reports `done/total` against a meaningful denominator.
        work_queue: list[Paper] = [
            p for p in papers if force or not p.extracted
        ]
        if limit is not None:
            work_queue = work_queue[:limit]
        total_work = len(work_queue)

        skipped_before = [p for p in papers if not (force or not p.extracted)]
        for p in skipped_before:
            results.append(ExtractionResult(paper_id=p.id, skipped=True))

        client = None
        if work_queue:
            try:
                import anthropic
                client = anthropic.Anthropic()
            except Exception as e:
                raise RuntimeError(
                    "anthropic client failed to initialize — set "
                    "ANTHROPIC_API_KEY or install `anthropic`. "
                    f"({type(e).__name__}: {e})"
                ) from e

        done = 0
        for p in work_queue:
            done += 1
            if progress_cb is not None:
                try:
                    progress_cb(done, total_work, p.title or p.arxiv_id)
                except Exception:  # noqa: BLE001 — never let callback break extraction
                    pass
            print(f"[extract] {p.arxiv_id}  {p.title[:72]}", flush=True)
            res = extract_entities(p, store=store, client=client, force=force)
            if res.error:
                print(f"[extract]   !! {res.error}", file=sys.stderr, flush=True)
            else:
                print(
                    f"[extract]   -> {len(res.node_ids)} nodes, "
                    f"{res.edge_count} edges, {res.elapsed_s:.1f}s",
                    flush=True,
                )
            results.append(res)
    finally:
        if owns_store:
            store.close()
    return results


# ── CLI entry (invoked via `sculpt kg extract --all`) ───────────────────────
def paper_ids_from_seeds(
    seeds_path: Path | str,
    *,
    tier: str | None = None,
    tag: str | None = None,
) -> set[str]:
    """Select exact Paper IDs from a structured campaign seeds file."""
    import yaml
    from sculptor.kg.ingest import _normalize_arxiv_id

    doc = yaml.safe_load(Path(seeds_path).read_text(encoding="utf-8")) or {}
    entries = doc.get("papers") or []
    if not isinstance(entries, list):
        raise ValueError("seeds `papers` must be a list")
    wanted_tier = tier.upper() if tier else None
    wanted_tag = tag.lower() if tag else None
    out: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            if not wanted_tier and not wanted_tag:
                out.add(make_paper_id(_normalize_arxiv_id(entry)))
            continue
        if not isinstance(entry, dict):
            continue
        if wanted_tier and str(entry.get("tier") or "").upper() != wanted_tier:
            continue
        entry_tags = {str(t).lower() for t in (entry.get("tags") or [])}
        if wanted_tag and wanted_tag not in entry_tags:
            continue
        aid = entry.get("arxiv_id") or entry.get("id")
        if aid:
            out.add(make_paper_id(_normalize_arxiv_id(str(aid))))
    return out


def cli_extract_all(
    store_path: Path | None,
    force: bool,
    limit: int | None,
    print_one: bool,
    *,
    paper_ids: set[str] | None = None,
    tier: str | None = None,
    tags: set[str] | None = None,
) -> int:
    store = SculptorKG(store_path) if store_path else SculptorKG()
    try:
        results = extract_all(
            store=store, force=force, limit=limit,
            paper_ids=paper_ids, tier=tier, tags=tags)
    finally:
        store.close()

    n_ok = sum(1 for r in results if not r.skipped and not r.error)
    n_skip = sum(1 for r in results if r.skipped)
    n_err = sum(1 for r in results if r.error)
    print(f"[extract] done. extracted={n_ok} skipped={n_skip} errors={n_err}")

    if print_one:
        for r in results:
            if r.raw_payload and not r.error:
                print(f"\n[extract] sample payload for {r.paper_id}:")
                print(json.dumps(r.raw_payload, indent=2, sort_keys=True))
                break

    return 0 if n_err == 0 else 1
