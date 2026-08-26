"""Thin wrappers over sculptor.kg for UI endpoints.

Everything here opens a fresh SculptorKG against the resolved KG path
— the UI must never share a store handle across requests (sqlite's
default threading mode is single-threaded per connection).

**M7 Phase 1**: the default KG path is user-wide shared at
`~/.local/share/sculptor/kg/graph.db`, seeded once via the bundled
pre-extracted DB. Override globally via the `RS_KG_PATH` env var.

**§Phase-0 hardening 2026-07-18**: the legacy per-project
`<project>/kg/graph.db` fallback is GONE — it was the same silent
graph-fragmentation trap sculptor's `default_db_path` removed on
2026-07-03 (loop 5a): a leftover project-local DB siloed that
project's diagnoses AND its run-case writes away from the shared
graph, and because `run_manager` exports this resolution as
`SCULPTOR_KG_PATH` to spawned runs, the split propagated to training.
A leftover legacy file now only triggers a once-per-project warning
pointing at `sculpt kg merge`.

Used by routes/kg.py.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

#: Legacy per-project DBs we already warned about this process — the
#: resolver runs on every request; one log line per project is enough.
_warned_legacy_dbs: set[Path] = set()

from backend.models.kg import (
    KGEntitySummary,
    KGStats,
    PaperDetail,
    PaperEntities,
    PaperSummary,
    ResearchCapabilitySummary,
    TechniqueSummary,
)


class KGIntegrityError(RuntimeError):
    """A reviewed graph claim is missing its required unambiguous status."""


def shared_kg_db_path() -> Path:
    """Resolve the user-wide shared KG path.

    Precedence (highest first):
      1. `$RS_KG_PATH` env var (tests + CI override).
      2. `$SCULPTOR_KG_PATH` env var (legacy sculptor-side alias).
      3. Default: `~/.local/share/sculptor/kg/graph.db`.

    Must agree with `sculptor.kg.store.shared_db_path()` on the
    default — duplicated so backend callers don't need a sculptor
    import just to resolve the path.
    """
    env = os.environ.get("RS_KG_PATH") or os.environ.get("SCULPTOR_KG_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (
        Path.home() / ".local" / "share" / "sculptor" / "kg" / "graph.db"
    ).resolve()


def project_kg_db_path(project_dir: Path) -> Path:
    """Resolve the KG DB path for a given project: ALWAYS the shared
    graph (or its env override) — one graph, one path, same contract as
    `sculptor.kg.store.default_db_path()`.

    A leftover legacy `<project>/kg/graph.db` is never opened; it gets
    a once-per-project warning pointing at `sculpt kg merge <path>`
    (additive-only — nothing in it can clobber the shared graph).
    """
    legacy = (project_dir / "kg" / "graph.db").resolve()
    shared = shared_kg_db_path()
    if legacy.is_file() and legacy != shared and legacy not in _warned_legacy_dbs:
        _warned_legacy_dbs.add(legacy)
        log.warning(
            "ignoring legacy per-project KG at %s — using the shared graph "
            "at %s. Merge it with `sculpt kg merge %s` (or delete it) to "
            "silence this.", legacy, shared, legacy)
    shared.parent.mkdir(parents=True, exist_ok=True)
    return shared


def _open_store(project_dir: Path):
    # Defer import so test fixtures can run without sculptor loaded.
    from sculptor.kg.store import SculptorKG

    db_path = project_kg_db_path(project_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SculptorKG(db_path)


def pdfs_dir(project_dir: Path) -> Path:
    """PDF cache next to the RESOLVED DB (matching `ingest._pdfs_dir`,
    which roots pdfs at `store.db_path.parent / "pdfs"`). The old
    `<project>/kg/pdfs` return value pointed at a directory the shared
    ingest never writes, so every `has_pdf` flag rendered False."""
    return project_kg_db_path(project_dir).parent / "pdfs"


def _paper_pdf_path(project_dir: Path, arxiv_id: str) -> Path:
    # Match sculptor.kg.ingest._safe_pdf_name for old-style IDs such as
    # hep-th/9901001; treating the slash as a directory made has_pdf false.
    return pdfs_dir(project_dir) / f"{arxiv_id.replace('/', '_')}.pdf"


# ── reads ─────────────────────────────────────────────────────────────
def list_papers(
    project_dir: Path,
    *,
    extracted: Optional[bool] = None,
    search: Optional[str] = None,
) -> list[PaperSummary]:
    with _open_store(project_dir) as kg:
        # Defer import so test fixtures can run without sculptor loaded.
        from sculptor.kg.schema import Paper

        nodes = kg.find_nodes(kind=Paper.kind)
        out: list[PaperSummary] = []
        for p in nodes:
            if extracted is not None and bool(p.extracted) != extracted:
                continue
            if search:
                needle = search.lower()
                hay = (
                    f"{p.title} {' '.join(p.authors or [])} {p.arxiv_id} "
                    f"{getattr(p, 'abstract', '')} "
                    f"{getattr(p, 'rationale', '')} "
                    f"{' '.join(getattr(p, 'tags', []) or [])} "
                    f"{_capability_search_text(kg, p.id)}"
                ).lower()
                if needle not in hay:
                    continue
            out.append(
                PaperSummary(
                    id=p.id,
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    authors=list(p.authors or []),
                    year=p.year,
                    extracted=bool(p.extracted),
                    has_pdf=_paper_pdf_path(project_dir, p.arxiv_id).is_file(),
                )
            )
        out.sort(key=lambda s: (-(s.year or 0), s.arxiv_id))
        return out


def get_paper_detail(
    project_dir: Path, arxiv_id: str
) -> Optional[PaperDetail]:
    with _open_store(project_dir) as kg:
        from sculptor.kg.schema import make_paper_id

        paper_id = make_paper_id(arxiv_id)
        paper = kg.get_node(paper_id)
        if paper is None:
            return None

        entities = _entities_for_paper(kg, paper_id)
        capabilities = _capabilities_for_paper(kg, paper_id)

        from datetime import datetime, timezone

        ingested_at = None
        try:
            ingested_at = datetime.fromtimestamp(
                float(getattr(paper, "ingested_at", 0) or 0), tz=timezone.utc
            )
        except (TypeError, ValueError):
            ingested_at = None

        return PaperDetail(
            id=paper.id,
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=list(paper.authors or []),
            year=paper.year,
            extracted=bool(paper.extracted),
            has_pdf=_paper_pdf_path(project_dir, paper.arxiv_id).is_file(),
            abstract=getattr(paper, "abstract", "") or "",
            conclusion_text=getattr(paper, "conclusion_text", "") or "",
            rationale=getattr(paper, "rationale", "") or "",
            tags=list(getattr(paper, "tags", []) or []),
            source_url=getattr(paper, "source_url", "") or "",
            provenance=getattr(paper, "provenance", "") or "",
            pdf_available=_paper_pdf_path(project_dir, paper.arxiv_id).is_file(),
            ingested_at=ingested_at,
            entities=entities,
            capabilities=capabilities,
        )


def _entities_for_paper(kg: Any, paper_id: str) -> PaperEntities:
    from sculptor.kg.schema import Relation

    out = PaperEntities()
    by_kind: dict[str, list[KGEntitySummary]] = {
        "Technique": [],
        "FailureMode": [],
        "RewardComponent": [],
        "Environment": [],
    }
    # Out-edges from the paper: INTRODUCES → (Technique|FailureMode|
    # RewardComponent), EVALUATES_ON → Environment.
    for relation in (Relation.INTRODUCES, Relation.EVALUATES_ON):
        for edge, other_id in kg.neighbors(
            paper_id, relation=relation, direction="out"
        ):
            node = kg.get_node(other_id)
            if node is None:
                continue
            kind = type(node).kind
            if kind in by_kind:
                by_kind[kind].append(
                    KGEntitySummary(
                        id=node.id,
                        kind=kind,  # type: ignore[arg-type]
                        name=getattr(node, "name", node.id),
                        description=getattr(node, "description", "") or "",
                    )
                )

    # Dedupe by id, preserving insertion order.
    def _dedup(xs: list[KGEntitySummary]) -> list[KGEntitySummary]:
        seen: set[str] = set()
        kept: list[KGEntitySummary] = []
        for x in xs:
            if x.id in seen:
                continue
            seen.add(x.id)
            kept.append(x)
        return kept

    out.techniques = _dedup(by_kind["Technique"])
    out.failure_modes = _dedup(by_kind["FailureMode"])
    out.reward_components = _dedup(by_kind["RewardComponent"])
    out.environments = _dedup(by_kind["Environment"])
    return out


def _capability_search_text(kg: Any, paper_id: str) -> str:
    """Plain lexical projection of reviewed claims for paper-list search.

    This intentionally does not infer a product status. Status integrity is
    enforced by the detail view; list search remains available even if an
    unrelated old graph row needs repair.
    """

    from sculptor.kg.schema import Relation, ResearchCapability

    chunks: list[str] = []
    for _edge, other_id in kg.neighbors(
        paper_id, relation=Relation.GROUNDS_CAPABILITY, direction="out"
    ):
        node = kg.get_node(other_id)
        if not isinstance(node, ResearchCapability):
            continue
        chunks.extend((node.name, node.description, node.scope))
        chunks.append(json.dumps(node.parameters, ensure_ascii=False, default=str))
    return " ".join(chunks)


def _capabilities_for_paper(
    kg: Any, paper_id: str
) -> list[ResearchCapabilitySummary]:
    """Return source-pinned mechanisms, failing closed on status ambiguity."""

    from sculptor.kg.schema import (
        ImplementationStatus,
        Relation,
        ResearchCapability,
    )

    out: list[ResearchCapabilitySummary] = []
    for grounding_edge, capability_id in kg.neighbors(
        paper_id, relation=Relation.GROUNDS_CAPABILITY, direction="out"
    ):
        capability = kg.get_node(capability_id)
        if not isinstance(capability, ResearchCapability):
            raise KGIntegrityError(
                f"{paper_id} grounds non-capability node {capability_id!r}"
            )
        status_edges = kg.neighbors(
            capability.id,
            relation=Relation.HAS_IMPLEMENTATION_STATUS,
            direction="out",
        )
        if len(status_edges) != 1:
            raise KGIntegrityError(
                f"{capability.id} must have exactly one implementation "
                f"status edge; found {len(status_edges)}"
            )
        status_node = kg.get_node(status_edges[0][1])
        if not isinstance(status_node, ImplementationStatus):
            raise KGIntegrityError(
                f"{capability.id} status target {status_edges[0][1]!r} is "
                "not an ImplementationStatus"
            )
        if status_node.status not in {
            "implemented", "metadata_only", "unsupported"
        }:
            raise KGIntegrityError(
                f"{capability.id} has unknown status {status_node.status!r}"
            )
        evidence = grounding_edge.data or {}
        out.append(
            ResearchCapabilitySummary(
                id=capability.id,
                name=capability.name,
                description=capability.description,
                scope=capability.scope,
                parameters=dict(capability.parameters or {}),
                implementation_status=status_node.status,
                status_definition=status_node.definition,
                code_evidence=list(capability.code_evidence or []),
                provenance=capability.provenance,
                paper_role=str(evidence.get("paper_role") or ""),
                source_version=str(
                    evidence.get("paper_version")
                    or evidence.get("source_version")
                    or ""
                ),
                source_locator=str(evidence.get("source_locator") or ""),
            )
        )
    out.sort(key=lambda item: item.name.casefold())
    return out


def list_techniques(project_dir: Path) -> list[TechniqueSummary]:
    with _open_store(project_dir) as kg:
        from sculptor.kg.schema import Relation, Technique

        techniques = kg.find_nodes(kind=Technique.kind)
        out: list[TechniqueSummary] = []
        for t in techniques:
            addresses = kg.neighbors(
                t.id, relation=Relation.ADDRESSES, direction="out"
            )
            introducers = [
                src_id for edge, src_id in kg.neighbors(
                    t.id, relation=Relation.INTRODUCES, direction="in"
                )
            ]
            out.append(
                TechniqueSummary(
                    id=t.id,
                    name=t.name,
                    description=t.description,
                    tags=list(t.tags or []),
                    n_addresses=len(addresses),
                    introduced_by=[
                        s.replace("paper:", "") for s in introducers
                    ],
                )
            )
        out.sort(key=lambda s: s.name.lower())
        return out


def get_stats(project_dir: Path) -> KGStats:
    with _open_store(project_dir) as kg:
        s = kg.stats()
        nodes = s.get("nodes_by_kind", {})
        return KGStats(
            papers=int(nodes.get("Paper", 0)),
            techniques=int(nodes.get("Technique", 0)),
            failure_modes=int(nodes.get("FailureMode", 0)),
            reward_components=int(nodes.get("RewardComponent", 0)),
            environments=int(nodes.get("Environment", 0)),
            results=int(nodes.get("Result", 0)),
            run_cases=int(nodes.get("RunCase", 0)),
            edges=int(s.get("total_edges", 0)),
            embeddings=int(s.get("total_embeddings", 0)),
        )
