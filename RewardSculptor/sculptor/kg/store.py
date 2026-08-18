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
Default DB path: the user-wide shared graph at
`~/.local/share/sculptor/kg/graph.db` (see `default_db_path`). Override per
call via the constructor, or globally via `SCULPTOR_KG_PATH` / `RS_KG_PATH`.
The parent directory is created on first use.

Embedding freshness (§Phase-0 hardening 2026-07-18)
---------------------------------------------------
`node_embeddings` carries a `text_hash` of the EXACT text that was embedded.
Callers that maintain lazy embedding pools (kg/query.py, kg/cases.py) pass
the canonical text into `set_embedding` and use `embedding_status` to detect
when a node's text changed after its vector was cached (extraction merges
enrich descriptions; case verdicts are re-attributed on resume) — a stale
vector is re-embedded instead of silently serving the old text's geometry.
Legacy rows (hash NULL) are trusted once and backfilled via
`backfill_embedding_hash`; `sculpt kg doctor --reembed-all` forces a full
refresh when that one-time trust is not wanted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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


log = logging.getLogger(__name__)

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
      1. `$RS_KG_PATH` if set (application-wide/backend override).
      2. `$SCULPTOR_KG_PATH` if set (legacy sculptor-side alias).
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
    # Keep this order identical to reward-sculptor-ui's resolver. If both
    # aliases are present, disagreeing precedence would make in-process UI
    # reads and default SculptorKG() writers open different databases.
    env = os.environ.get(BACKEND_ENV_VAR) or os.environ.get(ENV_VAR)
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
    text_hash  TEXT,
    PRIMARY KEY (node_id, model)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON node_embeddings(model);
"""


def embedding_text_hash(text: str) -> str:
    """Canonical hash of the exact text a vector was computed from."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _merge_corroborating_edge_data(
    destination_raw: str, source_raw: str,
) -> str | None:
    """Union paper-level claim supports without overwriting destination data.

    Edge JSON has an outer serialization envelope (``data`` +
    ``created_at``). Only claim provenance inside the nested data mapping is
    merged. Non-claim collisions remain destination-wins.
    """
    try:
        destination = json.loads(destination_raw)
        source = json.loads(source_raw)
    except (TypeError, ValueError):
        return None
    dst_data = destination.get("data")
    src_data = source.get("data")
    if not isinstance(dst_data, dict) or not isinstance(src_data, dict):
        return None

    def supports(data: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in data.get("supports", []) or []:
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("source_paper_id") or "")
            if paper_id:
                out[paper_id] = str(item.get("evidence") or "")
        paper_id = str(data.get("source_paper_id") or "")
        if paper_id:
            out.setdefault(paper_id, str(data.get("evidence") or ""))
        return out

    dst_supports = supports(dst_data)
    src_supports = supports(src_data)
    if not dst_supports and not src_supports:
        return None
    combined = dict(dst_supports)
    for paper_id, evidence in src_supports.items():
        if paper_id not in combined or (not combined[paper_id] and evidence):
            combined[paper_id] = evidence
    merged_data = dict(dst_data)
    merged_data["supports"] = [
        {"source_paper_id": paper_id, "evidence": evidence}
        for paper_id, evidence in sorted(combined.items())
    ]
    if not merged_data.get("source_paper_id") and src_data.get("source_paper_id"):
        merged_data["source_paper_id"] = src_data["source_paper_id"]
        merged_data["evidence"] = src_data.get("evidence", "")
    if merged_data == dst_data:
        return None
    destination["data"] = merged_data
    return json.dumps(destination, default=str)


def merge_stores(src_path: Path | str, dst: "SculptorKG") -> dict[str, int]:
    """Merge a stray/legacy KG database INTO `dst` (the shared graph).

    Additive-only, never destructive to `dst`:
      * nodes copied ONLY when the id is absent in dst — a legacy stub
        (e.g. a diagnoser-flagged FailureMode with a placeholder
        description) must never clobber the shared graph's richer
        paper-derived node of the same id;
      * edges: new claims are inserted and corroborating paper supports are
        unioned on composite-key collisions;
      * embeddings copied only when (node_id, model) is absent and the source
        row carries a text hash. Unhashed legacy vectors are never portable:
        their source text is unknowable, so lazy embedding must rebuild them.

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
    # An embedding can be copied safely only when the destination node is
    # byte-identical to the source node (including a newly copied node).
    # Copying a legacy vector onto a richer same-id destination node and then
    # trust-once stamping its NULL hash permanently mislabeled stale geometry
    # as fresh in the live graph.
    embedding_eligible_ids: set[str] = set()
    try:
        src_embedding_cols = {
            row["name"]
            for row in src.execute("PRAGMA table_info(node_embeddings)")
        }
        has_text_hash = "text_hash" in src_embedding_cols
        with dst._tx() as cx:
            for row in src.execute("SELECT id, kind, data FROM nodes"):
                existing = cx.execute(
                    "SELECT kind, data FROM nodes WHERE id = ?", (row["id"],)
                ).fetchone()
                cur = cx.execute(
                    "INSERT OR IGNORE INTO nodes (id, kind, data) "
                    "VALUES (?, ?, ?)",
                    (row["id"], row["kind"], row["data"]))
                if cur.rowcount:
                    counts["nodes"] += 1
                    embedding_eligible_ids.add(row["id"])
                else:
                    counts["nodes_skipped"] += 1
                    if (existing is not None
                            and existing["kind"] == row["kind"]
                            and existing["data"] == row["data"]):
                        embedding_eligible_ids.add(row["id"])
            for row in src.execute(
                    "SELECT src, dst, relation, data FROM edges"):
                existing_edge = cx.execute(
                    "SELECT data FROM edges "
                    "WHERE src = ? AND dst = ? AND relation = ?",
                    (row["src"], row["dst"], row["relation"]),
                ).fetchone()
                if existing_edge is None:
                    cx.execute(
                        "INSERT INTO edges (src, dst, relation, data) "
                        "VALUES (?, ?, ?, ?)",
                        (row["src"], row["dst"], row["relation"], row["data"]))
                    counts["edges"] += 1
                else:
                    merged = _merge_corroborating_edge_data(
                        existing_edge["data"], row["data"])
                    if merged is not None:
                        cx.execute(
                            "UPDATE edges SET data = ? "
                            "WHERE src = ? AND dst = ? AND relation = ?",
                            (merged, row["src"], row["dst"], row["relation"]))
                        counts["edges"] += 1
            if src_embedding_cols:  # very old stores may predate this table
                emb_sql = (
                    "SELECT node_id, model, dim, vector, updated_at, text_hash "
                    "FROM node_embeddings" if has_text_hash else
                    "SELECT node_id, model, dim, vector, updated_at, "
                    "NULL AS text_hash FROM node_embeddings"
                )
                for row in src.execute(emb_sql):
                    if (row["node_id"] not in embedding_eligible_ids
                            or not row["text_hash"]):
                        continue
                    cur = cx.execute(
                        "INSERT OR IGNORE INTO node_embeddings "
                        "(node_id, model, dim, vector, updated_at, text_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (row["node_id"], row["model"], row["dim"],
                         row["vector"], row["updated_at"], row["text_hash"]))
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
        self._transaction_depth = 0
        self._conn.executescript(_SCHEMA_SQL)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """In-place additive migrations for DBs created before a column
        existed. CREATE TABLE IF NOT EXISTS never alters an existing
        table, so new columns must be added here. Additive-only — never
        drops or rewrites data."""
        cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(node_embeddings)")
        }
        if "text_hash" not in cols:
            self._conn.execute(
                "ALTER TABLE node_embeddings ADD COLUMN text_hash TEXT")
        # Lexical index over paper BODIES. Paper retrieval ranks on
        # `title + abstract + rationale` only, so a paper whose body answers a
        # question but whose abstract never names the topic was unfindable —
        # and extraction reads at most 28K chars, so most bodies were never
        # even summarized into the graph. This index is the recall path over
        # the other ~55%. FTS5 is optional in a SQLite build; degrade to
        # "no lexical recall" rather than refusing to open the graph.
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS paper_fulltext "
                "USING fts5(node_id UNINDEXED, arxiv_id UNINDEXED, body)")
            self._fts_ok = True
        except sqlite3.OperationalError:  # pragma: no cover — build-dependent
            self._fts_ok = False

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
        if self._transaction_depth > 0:
            yield self._conn
            return
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @contextmanager
    def transaction(self) -> Iterator["SculptorKG"]:
        """Atomically group normal ``add_node``/``add_edge`` operations.

        Store methods historically committed each write independently.  This
        public boundary keeps that behavior for ordinary callers while letting
        multi-edge scientific attestations publish all-or-nothing.  Nested
        transactions join the outer unit instead of committing early.
        """
        outermost = self._transaction_depth == 0
        if outermost:
            self._conn.execute("BEGIN IMMEDIATE")
        self._transaction_depth += 1
        try:
            yield self
        except BaseException:
            self._transaction_depth -= 1
            if outermost:
                self._conn.rollback()
            raise
        else:
            self._transaction_depth -= 1
            if outermost:
                self._conn.commit()

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
            edge = self._row_to_edge_tolerant(row)
            if edge is None:
                continue
            other = edge.dst if edge.src == node_id else edge.src
            out.append((edge, other))
        return out

    def all_edges(self) -> Iterator[Edge]:
        """Yield every edge in the graph. Cheap — two-index table scan."""
        rows = self._conn.execute(
            "SELECT src, dst, relation, data FROM edges").fetchall()
        for row in rows:
            edge = self._row_to_edge_tolerant(row)
            if edge is not None:
                yield edge

    def _row_to_edge_tolerant(self, row: sqlite3.Row) -> Edge | None:
        """Deserialize one edge row; skip (with a once-per-relation warning)
        rows whose relation string is unknown to this code version instead
        of letting ONE such row abort a whole neighbors()/all_edges() scan.
        A newer schema writing a new Relation must not brick older readers'
        retrieval — the unknown edges are simply invisible to them."""
        try:
            return row_to_edge(
                row["src"], row["dst"], row["relation"],
                json.loads(row["data"]))
        except Exception:
            rel = str(row["relation"])
            warned = getattr(self, "_warned_relations_seen", None)
            if warned is None:
                warned = set()
                self._warned_relations_seen = warned
            if rel not in warned:
                warned.add(rel)
                log.warning(
                    "kg.store: skipping edge(s) with unknown/undecodable "
                    "relation %r (src=%r) — written by a newer schema or "
                    "corrupt; run `sculpt kg doctor` to enumerate",
                    rel, str(row["src"]))
            return None

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
    def set_embedding(
        self,
        node_id: str,
        model: str,
        vector: "np.ndarray",
        *,
        text: str | None = None,
    ) -> None:
        """Store (or replace) a node's embedding for a given model.

        `text`: the EXACT text the vector was computed from. When given,
        its hash is stored alongside so `embedding_status` can detect
        staleness after the node's text changes. Omitting it (legacy
        callers, tests seeding synthetic vectors) stores a NULL hash —
        treated as trusted-once by `embedding_status`."""
        import numpy as np

        v = np.asarray(vector, dtype=np.float32).ravel()
        blob = v.tobytes()
        th = embedding_text_hash(text) if text is not None else None
        with self._tx() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO node_embeddings "
                "(node_id, model, dim, vector, updated_at, text_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, model, int(v.shape[0]), blob, time.time(), th),
            )

    def embedding_status(self, node_id: str, model: str, text: str) -> str:
        """Freshness of a cached embedding vs `text` (the node's CURRENT
        canonical embed-text). One of:
          * "missing"  — no cached vector;
          * "fresh"    — cached hash matches `text`'s hash;
          * "stale"    — cached hash present and DIFFERENT (text changed);
          * "unhashed" — cached vector from before hash tracking (or
            seeded without text). Callers should `backfill_embedding_hash`
            (trust-once) or re-embed, per their policy."""
        row = self._conn.execute(
            "SELECT text_hash FROM node_embeddings "
            "WHERE node_id = ? AND model = ?",
            (node_id, model),
        ).fetchone()
        if row is None:
            return "missing"
        if row["text_hash"] is None:
            return "unhashed"
        return "fresh" if row["text_hash"] == embedding_text_hash(text) else "stale"

    def backfill_embedding_hash(self, node_id: str, model: str, text: str) -> None:
        """Stamp `text`'s hash onto an existing embedding row WITHOUT
        re-embedding — the trust-once migration for pre-hash rows (we
        cannot know what text produced the old vector; from this point on
        changes are tracked). No-op when the row is absent."""
        with self._tx() as cx:
            cx.execute(
                "UPDATE node_embeddings SET text_hash = ? "
                "WHERE node_id = ? AND model = ? AND text_hash IS NULL",
                (embedding_text_hash(text), node_id, model),
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

    # ── full-text (lexical) index over paper bodies ─────────────────────────
    def index_paper_fulltext(self, node_id: str, arxiv_id: str, body: str) -> bool:
        """Index one paper body for lexical search. Replaces any prior row for
        `node_id`, so re-indexing is idempotent. False when FTS5 is missing."""
        if not getattr(self, "_fts_ok", False) or not body:
            return False
        with self._tx() as c:
            c.execute("DELETE FROM paper_fulltext WHERE node_id = ?", (node_id,))
            c.execute(
                "INSERT INTO paper_fulltext(node_id, arxiv_id, body) "
                "VALUES (?, ?, ?)", (node_id, str(arxiv_id or ""), body))
        return True

    def search_paper_fulltext(
        self, query: str, *, limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Rank paper node_ids by BM25 over their bodies, best first.

        Returns `(node_id, score)` with score > 0 and larger = better (bm25()
        is negated: SQLite returns lower-is-better). Empty when FTS5 is
        unavailable, the query is blank, or the query is not valid FTS syntax —
        a user's free-text question is not guaranteed to parse, and a search
        that raises is worse than one that returns nothing.
        """
        if not getattr(self, "_fts_ok", False):
            return []
        terms = [t for t in re.findall(r"[A-Za-z0-9_]+", query or "") if len(t) > 2]
        if not terms:
            return []
        # OR the terms: a body matching any of them is a candidate worth
        # ranking. Quoting each defuses FTS operators in user text.
        expr = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self._conn.execute(
                "SELECT node_id, bm25(paper_fulltext) AS score FROM paper_fulltext "
                "WHERE paper_fulltext MATCH ? ORDER BY score LIMIT ?",
                (expr, int(limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(r["node_id"]), -float(r["score"])) for r in rows]

    def count_paper_fulltext(self) -> int:
        if not getattr(self, "_fts_ok", False):
            return 0
        return int(
            self._conn.execute("SELECT COUNT(*) FROM paper_fulltext").fetchone()[0])

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
