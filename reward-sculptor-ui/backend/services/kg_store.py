"""Thin wrappers over sculptor.kg for UI endpoints.

Everything here opens a fresh SculptorKG against the resolved KG path
— the UI must never share a store handle across requests (sqlite's
default threading mode is single-threaded per connection).

**M7 Phase 1**: the default KG path is now user-wide shared at
`~/.local/share/sculptor/kg/graph.db`, seeded once via the bundled
pre-extracted DB. Pre-existing per-project `<project>/kg/graph.db`
files are preserved in place (legacy fallback) so a user who had
iterated on a project-local KG before Phase 1 keeps seeing the same
graph. New projects default to the shared DB. Override globally via
the `RS_KG_PATH` env var.

Used by routes/kg.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from backend.models.kg import (
    KGEntitySummary,
    KGStats,
    PaperDetail,
    PaperEntities,
    PaperSummary,
    TechniqueSummary,
)


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
    """Resolve the KG DB path for a given project.

    Precedence (highest first):
      1. `shared_kg_db_path()` when a legacy per-project DB is ABSENT
         (the Phase-1 default — one graph per user).
      2. Legacy `<project>/kg/graph.db` if it already exists on disk
         (back-compat for pre-Phase-1 projects).

    Env-var overrides are folded into `shared_kg_db_path()`, so
    setting `RS_KG_PATH` redirects all projects at once — including
    newly-created ones.
    """
    legacy = project_dir / "kg" / "graph.db"
    if legacy.is_file():
        return legacy
    shared = shared_kg_db_path()
    shared.parent.mkdir(parents=True, exist_ok=True)
    return shared


def _open_store(project_dir: Path):
    # Defer import so test fixtures can run without sculptor loaded.
    from sculptor.kg.store import SculptorKG

    db_path = project_kg_db_path(project_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SculptorKG(db_path)


def pdfs_dir(project_dir: Path) -> Path:
    return project_dir / "kg" / "pdfs"


def _paper_pdf_path(project_dir: Path, arxiv_id: str) -> Path:
    return pdfs_dir(project_dir) / f"{arxiv_id}.pdf"


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
                    f"{p.title.lower()} "
                    f"{' '.join((p.authors or [])).lower()} "
                    f"{p.arxiv_id.lower()}"
                )
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
        from sculptor.kg.schema import make_paper_id, Relation

        paper_id = make_paper_id(arxiv_id)
        paper = kg.get_node(paper_id)
        if paper is None:
            return None

        entities = _entities_for_paper(kg, paper_id)

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
            pdf_available=_paper_pdf_path(project_dir, paper.arxiv_id).is_file(),
            ingested_at=ingested_at,
            entities=entities,
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
