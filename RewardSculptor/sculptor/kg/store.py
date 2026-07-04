"""sculptor/kg/store.py — SQLite-backed graph store for the Sculptor KG.

Shape
-----
Two tables, raw sqlite3, no ORM:

    nodes  (id TEXT PRIMARY KEY, kind TEXT NOT NULL, data TEXT NOT NULL)
    edges  (src TEXT NOT NULL, dst TEXT NOT NULL, relation TEXT NOT NULL,
            data TEXT NOT NULL, PRIMARY KEY (src, dst, relation))

`data` is a JSON blob; `nodes.id` and `nodes.kind` are hoisted for indexing.
Idempotent writes (INSERT OR REPLACE) so ingest can re-run without fighting
a unique constraint.

Path discovery
--------------
Default DB path: `<cwd>/kg/graph.db`. Override per call via the constructor,
or globally via the `SCULPTOR_KG_PATH` environment variable. The parent
directory is created on first use.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

from sculptor.kg.schema import (
    Edge,
    NODE_TYPES,
    Relation,
    edge_to_row,
    node_to_row,
    row_to_edge,
    row_to_node,
)


DEFAULT_RELATIVE_DB_PATH = Path("kg") / "graph.db"
ENV_VAR = "SCULPTOR_KG_PATH"
# Backend-facing alias. The reward-sculptor-ui backend sets RS_KG_PATH
# in tests + CI to point at a tmp DB without polluting the user's
# shared KG. Either env var wins over the default.
BACKEND_ENV_VAR = "RS_KG_PATH"


def shared_db_path() -> Path:
    """User-wide shared KG path. New default as of M7 Phase 1 — one
    graph seeded once + grown per prompt-time research, reused across
    every project the user creates."""
    return (
        Path.home() / ".local" / "share" / "sculptor" / "kg" / "graph.db"
    ).resolve()


def default_db_path() -> Path:
    """Resolve the default KG path.

    Resolution order:
      1. `$SCULPTOR_KG_PATH` if set (legacy override).
      2. `$RS_KG_PATH` if set (backend alias used by the UI test harness).
      3. The user-wide shared path: `~/.local/share/sculptor/kg/graph.db`.

    A cwd-relative `<cwd>/kg/graph.db` is NO LONGER honored (removed
    2026-07-03). The back-compat preference silently FRAGMENTED the
    graph by launch directory: the 2026-07 tuck-jump E2E ran its
    diagnoses against a 6-technique repo-local stub while the shared
    graph held 493 techniques + the whole run-case memory — the
    diagnoser was starved of 98 % of the KG and its cases were recorded
    into a silo no other entry point would ever read. One graph, one
    path; tests/CI isolate via the env vars.
    """
    env = os.environ.get(ENV_VAR) or os.environ.get(BACKEND_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    legacy = (Path.cwd() / DEFAULT_RELATIVE_DB_PATH).resolve()
    shared = shared_db_path()
    if legacy.is_file() and legacy != shared:
        import sys
        print(
            f"[kg] ignoring legacy per-directory DB at {legacy} — using the "
            f"shared graph at {shared}. Merge it with "
            f"`sculpt kg merge {legacy}` (or delete it) to silence this.",
            file=sys.stderr, flush=True,
        )
    return shared


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    data       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);

CREATE TABLE IF NOT EXISTS edges (
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    relation   TEXT NOT NULL,
    data       TEXT NOT NULL,
    PRIMARY KEY (src, dst, relation)
);
CREATE INDEX IF NOT EXISTS idx_edges_src      ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst      ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);

CREATE TABLE IF NOT EXISTS node_embeddings (
    node_id    TEXT NOT NULL,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (node_id, model)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON node_embeddings(model);
"""


def merge_stores(src_path: Path | str, dst: "SculptorKG") -> dict[str, int]:
    """Merge a stray/legacy KG database INTO `dst` (the shared graph).

    Additive-only, never destructive to `dst`:
      * nodes copied ONLY when the id is absent in dst — a legacy stub
        (e.g. a diagnoser-flagged FailureMode with a placeholder
        description) must never clobber the shared graph's richer
        paper-derived node of the same id;
      * edges: INSERT OR IGNORE (composite PK dedupes);
      * embeddings copied only when (node_id, model) is absent.

    The source file is left untouched — the caller decides whether to
    rename/delete it. Returns counts: {nodes, edges, embeddings,
    nodes_skipped}."""
    src_path = Path(src_path).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    if src_path == dst.db_path:
        raise ValueError("source and destination are the same database")
    src = sqlite3.connect(str(src_path))
    src.row_factory = sqlite3.Row
    counts = {"nodes": 0, "edges": 0, "embeddings": 0, "nodes_skipped": 0}
    try:
        with dst._tx() as cx:
            for row in src.execute("SELECT id, kind, data FROM nodes"):
                cur = cx.execute(
                    "INSERT OR IGNORE INTO nodes (id, kind, data) "
                    "VALUES (?, ?, ?)",
                    (row["id"], row["kind"], row["data"]))
                if cur.rowcount:
                    counts["nodes"] += 1
                else:
                    counts["nodes_skipped"] += 1
            for row in src.execute(
                    "SELECT src, dst, relation, data FROM edges"):
                cur = cx.execute(
                    "INSERT OR IGNORE INTO edges (src, dst, relation, data) "
                    "VALUES (?, ?, ?, ?)",
                    (row["src"], row["dst"], row["relation"], row["data"]))
                counts["edges"] += int(cur.rowcount or 0)
            for row in src.execute(
                    "SELECT node_id, model, dim, vector, updated_at "
                    "FROM node_embeddings"):
                cur = cx.execute(
                    "INSERT OR IGNORE INTO node_embeddings "
                    "(node_id, model, dim, vector, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (row["node_id"], row["model"], row["dim"],
                     row["vector"], row["updated_at"]))
                counts["embeddings"] += int(cur.rowcount or 0)
    finally:
        src.close()
    return counts


class SculptorKG:
    """Minimal sqlite-backed node/edge store."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path: Path = (
            Path(db_path).expanduser().resolve() if db_path else default_db_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ── nodes ───────────────────────────────────────────────────────────────
    def add_node(self, node: Any, *, upsert: bool = True) -> str:
        """Insert (or replace if upsert=True) a node. Returns its id."""
        node_id, kind, data = node_to_row(node)
        sql = (
            "INSERT OR REPLACE INTO nodes (id, kind, data) VALUES (?, ?, ?)"
            if upsert
            else "INSERT INTO nodes (id, kind, data) VALUES (?, ?, ?)"
        )
        with self._tx() as cx:
            cx.execute(sql, (node_id, kind, json.dumps(data, default=str)))
        return node_id

    def has_node(self, node_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return row is not None

    def get_node(self, node_id: str) -> Any | None:
        row = self._conn.execute(
            "SELECT id, kind, data FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return row_to_node(row["id"], row["kind"], json.loads(row["data"]))

    def find_nodes(self, *, kind: str | None = None, **filters: Any) -> list[Any]:
        """List nodes of a given kind, optionally filtered by `data.<key>`.

        Filter semantics: equality on the JSON blob field. For list fields we
        check membership (filter value ∈ list). For booleans and numbers we
        compare directly. None values are ignored (treated as "no filter").
        """
        sql = "SELECT id, kind, data FROM nodes"
        args: list[Any] = []
        if kind is not None:
            sql += " WHERE kind = ?"
            args.append(kind)
        rows = self._conn.execute(sql, args).fetchall()
        out: list[Any] = []
        for row in rows:
            data = json.loads(row["data"])
            if not _matches_filters(data, filters):
                continue
            try:
                out.append(row_to_node(row["id"], row["kind"], data))
            except Exception:
                # Defensively skip rows that no longer match any known class
                # (e.g. older node kinds dropped from NODE_TYPES).
                continue
        return out

    def delete_node(self, node_id: str) -> bool:
        with self._tx() as cx:
            cur = cx.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            cx.execute("DELETE FROM edges WHERE src = ? OR dst = ?", (node_id, node_id))
            cx.execute("DELETE FROM node_embeddings WHERE node_id = ?", (node_id,))
        return cur.rowcount > 0

    # ── edges ───────────────────────────────────────────────────────────────
    def add_edge(self, edge: Edge, *, upsert: bool = True) -> None:
        src, dst, rel, extra = edge_to_row(edge)
        sql = (
            "INSERT OR REPLACE INTO edges (src, dst, relation, data) VALUES (?, ?, ?, ?)"
            if upsert
            else "INSERT INTO edges (src, dst, relation, data) VALUES (?, ?, ?, ?)"
        )
        with self._tx() as cx:
            cx.execute(sql, (src, dst, rel, json.dumps(extra, default=str)))

    def neighbors(
        self,
        node_id: str,
        *,
        relation: Relation | str | None = None,
        direction: str = "out",
    ) -> list[tuple[Edge, str]]:
        """Return [(edge, other_node_id), ...] for edges touching `node_id`.

        direction: 'out' (node_id is src), 'in' (node_id is dst), 'both'.
        """
        rel_str = relation.value if isinstance(relation, Relation) else relation

        if direction == "out":
            sql = "SELECT * FROM edges WHERE src = ?"
            args: list[Any] = [node_id]
        elif direction == "in":
            sql = "SELECT * FROM edges WHERE dst = ?"
            args = [node_id]
        elif direction == "both":
            sql = "SELECT * FROM edges WHERE src = ? OR dst = ?"
            args = [node_id, node_id]
        else:
            raise ValueError(f"direction must be 'out', 'in', or 'both'; got {direction!r}")

        if rel_str is not None:
            sql += " AND relation = ?"
            args.append(rel_str)

        out: list[tuple[Edge, str]] = []
        for row in self._conn.execute(sql, args).fetchall():
            edge = row_to_edge(row["src"], row["dst"], row["relation"], json.loads(row["data"]))
            other = edge.dst if edge.src == node_id else edge.src
            out.append((edge, other))
        return out

    def all_edges(self) -> Iterator[Edge]:
        """Yield every edge in the graph. Cheap — two-index table scan."""
        rows = self._conn.execute(
            "SELECT src, dst, relation, data FROM edges").fetchall()
        for row in rows:
            yield row_to_edge(
                row["src"], row["dst"], row["relation"],
                json.loads(row["data"]))

    # ── counts / stats ──────────────────────────────────────────────────────
    def count_nodes(self, kind: str | None = None) -> int:
        if kind is None:
            r = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
        else:
            r = self._conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE kind = ?", (kind,)).fetchone()
        return int(r[0])

    def count_edges(self, relation: Relation | str | None = None) -> int:
        if relation is None:
            r = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()
            return int(r[0])
        rel_str = relation.value if isinstance(relation, Relation) else relation
        r = self._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE relation = ?", (rel_str,)).fetchone()
        return int(r[0])

    # ── embeddings ──────────────────────────────────────────────────────────
    def set_embedding(self, node_id: str, model: str, vector: "np.ndarray") -> None:
        """Store (or replace) a node's embedding for a given model."""
        import numpy as np

        v = np.asarray(vector, dtype=np.float32).ravel()
        blob = v.tobytes()
        with self._tx() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO node_embeddings "
                "(node_id, model, dim, vector, updated_at) VALUES (?, ?, ?, ?, ?)",
                (node_id, model, int(v.shape[0]), blob, time.time()),
            )

    def get_embedding(self, node_id: str, model: str) -> "np.ndarray | None":
        import numpy as np

        row = self._conn.execute(
            "SELECT dim, vector FROM node_embeddings WHERE node_id = ? AND model = ?",
            (node_id, model),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row["vector"], dtype=np.float32).reshape(int(row["dim"]))

    def iter_embeddings(self, model: str) -> Iterator[tuple[str, "np.ndarray"]]:
        import numpy as np

        rows = self._conn.execute(
            "SELECT node_id, dim, vector FROM node_embeddings WHERE model = ?",
            (model,),
        ).fetchall()
        for row in rows:
            vec = np.frombuffer(row["vector"], dtype=np.float32).reshape(int(row["dim"]))
            yield row["node_id"], vec

    def has_embedding(self, node_id: str, model: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM node_embeddings WHERE node_id = ? AND model = ?",
            (node_id, model),
        ).fetchone()
        return row is not None

    def count_embeddings(self, model: str | None = None) -> int:
        if model is None:
            r = self._conn.execute("SELECT COUNT(*) FROM node_embeddings").fetchone()
        else:
            r = self._conn.execute(
                "SELECT COUNT(*) FROM node_embeddings WHERE model = ?", (model,)
            ).fetchone()
        return int(r[0])

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "db_path": str(self.db_path),
            "total_nodes": self.count_nodes(),
            "total_edges": self.count_edges(),
            "total_embeddings": self.count_embeddings(),
            "nodes_by_kind": {},
            "edges_by_relation": {},
        }
        for kind in NODE_TYPES:
            n = self.count_nodes(kind=kind)
            if n:
                out["nodes_by_kind"][kind] = n
        for rel in Relation:
            n = self.count_edges(relation=rel)
            if n:
                out["edges_by_relation"][rel.value] = n
        return out


# ── helpers ─────────────────────────────────────────────────────────────────
def _matches_filters(data: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if value is None:
            continue
        if key not in data:
            return False
        field_val = data[key]
        if isinstance(field_val, list):
            if value not in field_val:
                return False
        else:
            if field_val != value:
                return False
    return True
