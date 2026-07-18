"""sculptor/kg/doctor.py — KG integrity checks + mechanical repairs.

§Phase-0 hardening 2026-07-18. The shared graph accretes from many
writers (seed ingest, LLM extraction, run-case memory, UI research jobs,
merges of legacy DBs) across sessions whose cwd / tmp layout differ — so
referential slack accumulates silently: edges to deleted nodes, embedding
rows for deleted nodes, papers whose PDF sidecar died with a /tmp, stale
vectors after description-enriching merges. None of these raise anywhere;
they just quietly degrade retrieval. `run_doctor` makes every one of them
visible in a single report, and `--fix` repairs the mechanical ones.

CLI:  sculpt kg doctor [--fix] [--reembed-all] [--no-network]

Repairs under `fix=True`:
  * delete embedding rows whose node no longer exists (orphans);
  * delete edges whose src/dst node no longer exists (dangling);
  * re-embed MISSING + STALE vectors for the three semantic pools
    (Technique / FailureMode / RunCase) and stamp trust-once hashes on
    pre-hash rows — via kg.query.ensure_embeddings;
  * (network) heal stub-titled papers (`ingest.heal_stub_titles`);
  * (network) heal dead full_text_path sidecars
    (`ingest.heal_dead_text_paths`).

`reembed_all=True` additionally drops every cached vector for the three
pools first — the escape hatch for embeddings whose pre-hash staleness is
unknowable (hash tracking only starts at the first post-upgrade scan).

Read-only by default: `run_doctor(fix=False)` never writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sculptor.kg.schema import NODE_TYPES, Relation
from sculptor.kg.store import SculptorKG


def _known_relations() -> set[str]:
    return {r.value for r in Relation}


def _semantic_pools(store: SculptorKG):
    """The three (kind, nodes, text_fn) semantic-retrieval pools, using the
    SAME canonical text builders the live query paths embed with."""
    from sculptor.kg.cases import _case_text
    from sculptor.kg.query import (
        failure_mode_embed_text,
        technique_embed_text,
    )

    return [
        ("Technique",
         store.find_nodes(kind="Technique"), technique_embed_text),
        ("FailureMode",
         store.find_nodes(kind="FailureMode"), failure_mode_embed_text),
        ("RunCase", store.find_nodes(kind="RunCase"), _case_text),
    ]


def run_doctor(
    store: SculptorKG | None = None,
    *,
    fix: bool = False,
    reembed_all: bool = False,
    network: bool = True,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run every integrity check; optionally repair. Returns the report.

    Report keys (counts unless noted):
      db_path, nodes, edges, embeddings,
      dangling_edges          — list[(src, relation, dst)]
      orphan_embeddings       — list[node_id]
      unknown_kind_nodes      — list[(id, kind)]
      unknown_relation_edges  — list[(src, relation, dst)]
      stub_titled_papers      — list[arxiv_id]
      dead_text_paths         — list[arxiv_id]
      unextracted_papers      — int
      embedding_pools         — {kind: {total, missing, stale, unhashed}}
      fixes                   — {action: outcome} (only when fix=True)
    """
    from sculptor.kg.query import EMBEDDING_MODEL

    model = model_name or EMBEDDING_MODEL
    owns_store = store is None
    store = store or SculptorKG()
    report: dict[str, Any] = {}
    try:
        cx = store._conn  # same-package access; doctor IS the store's audit
        report["db_path"] = str(store.db_path)
        report["nodes"] = store.count_nodes()
        report["edges"] = store.count_edges()
        report["embeddings"] = store.count_embeddings()

        report["dangling_edges"] = [
            (r["src"], r["relation"], r["dst"])
            for r in cx.execute(
                "SELECT e.src, e.relation, e.dst FROM edges e "
                "LEFT JOIN nodes s ON e.src = s.id "
                "LEFT JOIN nodes d ON e.dst = d.id "
                "WHERE s.id IS NULL OR d.id IS NULL")
        ]
        report["orphan_embeddings"] = [
            r["node_id"]
            for r in cx.execute(
                "SELECT DISTINCT ne.node_id FROM node_embeddings ne "
                "LEFT JOIN nodes n ON ne.node_id = n.id "
                "WHERE n.id IS NULL")
        ]
        known_kinds = set(NODE_TYPES)
        report["unknown_kind_nodes"] = [
            (r["id"], r["kind"])
            for r in cx.execute("SELECT id, kind FROM nodes")
            if r["kind"] not in known_kinds
        ]
        known_rels = _known_relations()
        report["unknown_relation_edges"] = [
            (r["src"], r["relation"], r["dst"])
            for r in cx.execute("SELECT src, relation, dst FROM edges")
            if r["relation"] not in known_rels
        ]

        stub_titles: list[str] = []
        dead_paths: list[str] = []
        unextracted = 0
        for paper in store.find_nodes(kind="Paper"):
            title = getattr(paper, "title", "") or ""
            aid = str(getattr(paper, "arxiv_id", "") or "")
            if isinstance(title, str) and title.startswith("arxiv:"):
                stub_titles.append(aid)
            path = getattr(paper, "full_text_path", None)
            if path and not Path(path).is_file():
                dead_paths.append(aid)
            if not getattr(paper, "extracted", False):
                unextracted += 1
        report["stub_titled_papers"] = stub_titles
        report["dead_text_paths"] = dead_paths
        report["unextracted_papers"] = unextracted

        pools: dict[str, dict[str, int]] = {}
        pool_specs = _semantic_pools(store)
        for kind, nodes, text_fn in pool_specs:
            counts = {"total": len(nodes), "missing": 0, "stale": 0,
                      "unhashed": 0}
            for n in nodes:
                text = text_fn(n)
                if not text:
                    continue
                status = store.embedding_status(n.id, model, text)
                if status in counts:
                    counts[status] = counts.get(status, 0) + 1
                elif status == "missing":
                    counts["missing"] += 1
            pools[kind] = counts
        report["embedding_pools"] = pools

        if not fix:
            return report

        fixes: dict[str, Any] = {}
        report["fixes"] = fixes

        if report["orphan_embeddings"]:
            with store._tx() as tx:
                tx.execute(
                    "DELETE FROM node_embeddings WHERE node_id IN "
                    "(SELECT ne.node_id FROM node_embeddings ne "
                    " LEFT JOIN nodes n ON ne.node_id = n.id "
                    " WHERE n.id IS NULL)")
            fixes["orphan_embeddings_deleted"] = len(
                report["orphan_embeddings"])

        if report["dangling_edges"]:
            with store._tx() as tx:
                tx.execute(
                    "DELETE FROM edges WHERE rowid IN "
                    "(SELECT e.rowid FROM edges e "
                    " LEFT JOIN nodes s ON e.src = s.id "
                    " LEFT JOIN nodes d ON e.dst = d.id "
                    " WHERE s.id IS NULL OR d.id IS NULL)")
            fixes["dangling_edges_deleted"] = len(report["dangling_edges"])

        if reembed_all:
            pool_ids = [
                n.id for _, nodes, _ in pool_specs for n in nodes]
            with store._tx() as tx:
                tx.executemany(
                    "DELETE FROM node_embeddings "
                    "WHERE node_id = ? AND model = ?",
                    [(nid, model) for nid in pool_ids])
            fixes["reembed_all_dropped"] = len(pool_ids)

        # Re-embed missing + stale (and, after reembed_all, everything).
        from sculptor.kg.query import ensure_embeddings

        reembedded: dict[str, int] = {}
        for kind, nodes, text_fn in pool_specs:
            before = store.count_embeddings(model)
            ensure_embeddings(store, nodes, text_fn, model, label=kind)
            reembedded[kind] = store.count_embeddings(model) - before
        fixes["embeddings_added_by_kind"] = reembedded

        if network and report["stub_titled_papers"]:
            from sculptor.kg.ingest import heal_stub_titles

            fixes["stub_titles"] = heal_stub_titles(store=store)
        if network and report["dead_text_paths"]:
            from sculptor.kg.ingest import heal_dead_text_paths

            fixes["dead_text_paths"] = heal_dead_text_paths(store=store)

        return report
    finally:
        if owns_store:
            store.close()


def format_report(report: dict[str, Any]) -> str:
    """Human-readable rendering for the CLI."""
    lines = [
        f"KG doctor — {report['db_path']}",
        f"  nodes={report['nodes']} edges={report['edges']} "
        f"embeddings={report['embeddings']}",
    ]

    def _issue(name: str, items: list, fmt=lambda x: str(x)) -> None:
        if not items:
            lines.append(f"  [ok] {name}: 0")
            return
        lines.append(f"  [!!] {name}: {len(items)}")
        for it in items[:8]:
            lines.append(f"        {fmt(it)}")
        if len(items) > 8:
            lines.append(f"        ... and {len(items) - 8} more")

    _issue("dangling edges", report["dangling_edges"],
           lambda e: f"{e[0]} -[{e[1]}]-> {e[2]}")
    _issue("orphan embeddings", report["orphan_embeddings"])
    _issue("unknown-kind nodes", report["unknown_kind_nodes"],
           lambda n: f"{n[0]} (kind={n[1]})")
    _issue("unknown-relation edges", report["unknown_relation_edges"],
           lambda e: f"{e[0]} -[{e[1]}]-> {e[2]}")
    _issue("stub-titled papers", report["stub_titled_papers"])
    _issue("dead full_text_path papers", report["dead_text_paths"])
    lines.append(f"  [--] unextracted papers: {report['unextracted_papers']}")
    for kind, c in report["embedding_pools"].items():
        flag = "ok" if not (c["missing"] or c["stale"]) else "!!"
        lines.append(
            f"  [{flag}] {kind} embeddings: total={c['total']} "
            f"missing={c['missing']} stale={c['stale']} "
            f"unhashed={c['unhashed']}")
    if "fixes" in report:
        lines.append("  fixes applied:")
        for k, v in report["fixes"].items():
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)
