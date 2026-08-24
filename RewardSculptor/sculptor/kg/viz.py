"""sculptor/kg/viz.py — interactive pyvis rendering of the KG.

`build_kg_html(store, out_path, provenance=None)` writes a self-contained
HTML file. Open it in a browser to explore the graph: nodes are colored by
type, active nodes (those whose target_term shows up in the current
project's provenance.json with `still_active=True`, plus the papers they
cite) wear a gold halo, and every edge is labeled with its relation.

CLI: `sculpt kg viz --config <config.toml> --out kg.html`

The `--config` is optional — without it, the viz renders the full KG
without any "active" highlighting.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sculptor.kg.schema import (
    ArtifactAttestation,
    Environment,
    FailureMode,
    ImplementationStatus,
    ModeExecutionArtifact,
    Paper,
    PolicyArtifact,
    ReferenceMotion,
    ResearchCapability,
    RewardComponent,
    Result,
    RobotEmbodiment,
    RunCase,
    SoftwareEnvironment,
    Technique,
    TrainingIteration,
    TrainingRun,
    WorldArtifact,
)
from sculptor.kg.store import SculptorKG


# Visual theme — dark background + saturated node colors so the graph pops
# on a presentation slide.
_NODE_COLORS = {
    "Paper":           {"background": "#4fc3f7", "border": "#1e88e5"},
    "Technique":       {"background": "#81c784", "border": "#43a047"},
    "FailureMode":     {"background": "#ef5350", "border": "#c62828"},
    "RewardComponent": {"background": "#ba68c8", "border": "#8e24aa"},
    "Environment":     {"background": "#ffb74d", "border": "#f57c00"},
    "Result":          {"background": "#90a4ae", "border": "#546e7a"},
    # §2026-07-03: the run-experience silo is a first-class visual citizen
    # (it was falling through to gray with a raw `case:...` id as label).
    "RunCase":         {"background": "#4dd0e1", "border": "#00acc1"},
    "ResearchCapability": {
        "background": "#64b5f6", "border": "#1565c0"},
    "ImplementationStatus": {
        "background": "#fff176", "border": "#f9a825"},
    # Immutable artifact and execution-lineage nodes. These previously fell
    # through to the anonymous gray fallback, which made a content hash, an
    # attestation, and an observed run visually indistinguishable.
    "PolicyArtifact":  {"background": "#26a69a", "border": "#00796b"},
    "ReferenceMotion": {"background": "#ab47bc", "border": "#7b1fa2"},
    "WorldArtifact":   {"background": "#ffa726", "border": "#ef6c00"},
    "ModeExecutionArtifact": {
        "background": "#ec407a", "border": "#ad1457"},
    "ArtifactAttestation": {
        "background": "#ffee58", "border": "#f9a825"},
    "RobotEmbodiment": {"background": "#66bb6a", "border": "#2e7d32"},
    "SoftwareEnvironment": {
        "background": "#78909c", "border": "#455a64"},
    "TrainingRun":     {"background": "#42a5f5", "border": "#1565c0"},
    "TrainingIteration": {
        "background": "#26c6da", "border": "#00838f"},
}
_ACTIVE_COLOR = {"background": "#ffd54f", "border": "#ffa000"}  # gold halo
_FALLBACK_COLOR = {"background": "#bdbdbd", "border": "#757575"}
_NODE_SHAPES = {
    "RunCase": "diamond",
    "PolicyArtifact": "box",
    "ReferenceMotion": "box",
    "WorldArtifact": "box",
    "ModeExecutionArtifact": "box",
    "ArtifactAttestation": "hexagon",
    "RobotEmbodiment": "ellipse",
    "SoftwareEnvironment": "database",
    "TrainingRun": "triangle",
    "TrainingIteration": "dot",
}
# RunCase borders encode the measured verdict at a glance.
_VERDICT_BORDER = {"helped": "#00e676", "regressed": "#ff1744"}
_VERDICT_GLYPH = {"helped": "✓", "regressed": "✗", "neutral": "=",
                  "unknown": "?"}

_EDGE_COLORS = {
    "INTRODUCES":    "#66bb6a",
    "ADDRESSES":     "#ef5350",
    "USES":          "#ba68c8",
    "EVALUATES_ON":  "#ffb74d",
    "CITES":         "#78909c",
    "REPORTS":       "#9ccc65",
    "IMPROVES_OVER": "#42a5f5",
    "GROUNDS_CAPABILITY": "#5c6bc0",
    "HAS_IMPLEMENTATION_STATUS": "#fdd835",
    "INSTANTIATES": "#ef5350",
    "INITIALIZED_FROM": "#26a69a",
    "TRACKS": "#ab47bc",
    "EXECUTES_IN": "#ffa726",
    "DECLARES_TARGET": "#9ccc65",
    "COMPATIBLE_WITH": "#66bb6a",
    "DERIVED_FROM": "#00897b",
    "ATTESTS": "#fdd835",
    "PRODUCED": "#42a5f5",
    "USES_MODE_EXECUTION": "#ec407a",
    "HAS_ITERATION": "#26c6da",
}
_DEFAULT_EDGE_COLOR = "#b0bec5"


@dataclass
class VizResult:
    out_path: Path
    n_nodes: int
    n_edges: int
    n_active_nodes: int


# ── Tooltip / label helpers ─────────────────────────────────────────────
def _short_hash(value: str | None) -> str:
    text = str(value or "")
    return text[:10] + ("…" if len(text) > 10 else "")


def _label_for(node: Any) -> str:
    """Short label shown on the node itself."""
    if isinstance(node, Paper):
        # arxiv_id is compact + recognizable
        return node.arxiv_id or node.title[:30]
    if isinstance(node, (Technique, FailureMode, RewardComponent, Environment)):
        name = getattr(node, "name", None) or node.id
        return str(name)[:40]
    if isinstance(node, Result):
        return f"{node.metric_name}={node.value}"
    if isinstance(node, RunCase):
        # "iter 12 ✗" — the iteration + verdict at a glance. The iter index
        # is the second-to-last id segment (case:<task>:<iter>:<nonce>).
        glyph = _VERDICT_GLYPH.get(node.verdict, "?")
        parts = node.id.split(":")
        it = parts[-2] if len(parts) >= 3 else "?"
        return f"iter {it} {glyph}"
    if isinstance(node, ResearchCapability):
        return node.name[:40]
    if isinstance(node, ImplementationStatus):
        return node.status.replace("_", " ")[:40]
    if isinstance(node, PolicyArtifact):
        return f"policy {_short_hash(node.sha256)}"
    if isinstance(node, ReferenceMotion):
        return f"motion {_short_hash(node.sha256)}"
    if isinstance(node, WorldArtifact):
        return f"world {_short_hash(node.sha256)}"
    if isinstance(node, ModeExecutionArtifact):
        return f"mode {node.clip_id[:28]}"
    if isinstance(node, ArtifactAttestation):
        return f"attestation {_short_hash(node.manifest_digest)}"
    if isinstance(node, RobotEmbodiment):
        return f"robot {node.slug[:32]}"
    if isinstance(node, SoftwareEnvironment):
        identity = node.code_commit or node.code_tree_digest or node.lock_digest
        return f"software {_short_hash(identity)}"
    if isinstance(node, TrainingRun):
        return f"{node.project[:20]} · {node.run_id[:18]}"
    if isinstance(node, TrainingIteration):
        return f"iter {node.iteration_index}"
    return str(getattr(node, "id", node))[:40]


def _tooltip_for(node: Any, active_reason: str | None = None) -> str:
    """Rich HTML tooltip shown on hover."""
    kind = type(node).__name__
    header = f"<b>{kind}</b>"
    rows: list[str] = []

    def _row(label: str, value: str) -> None:
        rows.append(
            f"<div style='margin:4px 0'>"
            f"<span style='color:#8fb5ff;'>{label}:</span> "
            f"{html.escape(value)}</div>")

    if isinstance(node, Paper):
        _row("title", str(node.title)[:200])
        if node.authors:
            authors = ", ".join(node.authors[:3]) + (
                ", ..." if len(node.authors) > 3 else "")
            _row("authors", authors)
        if node.year is not None:
            _row("year", str(node.year))
        _row("arxiv_id", str(node.arxiv_id))
        if node.abstract:
            abstract = node.abstract.replace("\n", " ").strip()
            _row("abstract", abstract[:400] + ("…" if len(abstract) > 400 else ""))
    elif isinstance(node, Technique):
        _row("name", str(node.name))
        if node.description:
            _row("description", str(node.description)[:400])
        if node.tags:
            _row("tags", ", ".join(node.tags))
    elif isinstance(node, FailureMode):
        _row("name", str(node.name))
        if node.description:
            _row("description", str(node.description)[:400])
        if node.symptoms:
            _row("symptoms", "; ".join(str(s) for s in node.symptoms)[:400])
        if node.environment_tag:
            _row("environment_tag", node.environment_tag)
    elif isinstance(node, RewardComponent):
        _row("name", str(node.name))
        if node.description:
            _row("description", str(node.description)[:400])
        if node.formula:
            _row("formula", str(node.formula)[:200])
        if node.hyperparameters:
            _row("hyperparameters",
                 json.dumps(node.hyperparameters)[:200])
    elif isinstance(node, Environment):
        _row("name", str(node.name))
        if node.description:
            _row("description", str(node.description)[:400])
        if node.tags:
            _row("tags", ", ".join(node.tags))
    elif isinstance(node, Result):
        _row("metric", f"{node.metric_name} = {node.value}")
        if node.environment_id:
            _row("env", node.environment_id)
        if node.notes:
            _row("notes", str(node.notes)[:200])
    elif isinstance(node, RunCase):
        _row("task", str(node.task)[:160])
        _row("symptom", str(node.symptom)[:160])
        if node.edits:
            _row("edits", "; ".join(node.edits)[:300])
        verdict_bits = [node.verdict]
        if node.fitness_delta is not None:
            verdict_bits.append(f"fitness Δ {node.fitness_delta:+.4f}")
        if node.progress_delta is not None:
            verdict_bits.append(f"progress Δ {node.progress_delta:+.4f}")
        _row("verdict", " | ".join(verdict_bits))
        if node.behavior:
            _row("behavior", ", ".join(
                f"{k}={v:g}" for k, v in node.behavior.items())[:300])
    elif isinstance(node, ResearchCapability):
        _row("name", node.name)
        _row("scope", node.scope)
        if node.description:
            _row("description", node.description[:400])
        if node.code_evidence:
            _row("code evidence", "; ".join(node.code_evidence)[:400])
    elif isinstance(node, ImplementationStatus):
        _row("status", node.status.replace("_", " "))
        _row("definition", node.definition[:400])
    elif isinstance(node, PolicyArtifact):
        _row("sha256", node.sha256)
        _row("format", node.artifact_format)
        if node.size_bytes is not None:
            _row("size bytes", str(node.size_bytes))
        if node.tensor_inventory_digest:
            _row("tensor inventory digest", node.tensor_inventory_digest)
        _row("provenance", node.provenance)
    elif isinstance(node, ReferenceMotion):
        _row("sha256", node.sha256)
        if node.fps is not None:
            _row("fps", str(node.fps))
        if node.frame_count is not None:
            _row("frames", str(node.frame_count))
        if node.joint_names:
            _row("ordered joints", ", ".join(node.joint_names))
        _row("provenance", node.provenance)
    elif isinstance(node, WorldArtifact):
        _row("sha256", node.sha256)
        _row("format", node.artifact_format or "not recorded")
        _row("provenance", node.provenance)
    elif isinstance(node, ModeExecutionArtifact):
        _row("bundle digest", node.bundle_digest)
        _row("reward sha256", node.reward_sha256)
        _row("robot", node.robot)
        _row("clip", f"{node.clip_id} / {node.clip_sha256}")
        _row("mode graph sha256", node.graph_sha256)
        _row("execution manifest", node.execution_manifest_digest)
        _row("world selection", node.selection_digest)
        _row("context refs digest", node.context_refs_digest)
        if node.context_refs:
            _row("context refs", json.dumps(
                node.context_refs, ensure_ascii=False, sort_keys=True))
        _row("provenance", node.provenance)
    elif isinstance(node, ArtifactAttestation):
        _row("manifest digest", node.manifest_digest)
        _row("trust status", node.trust_status)
        _row("source format", node.source_format)
        _row("admitted at", str(node.admitted_at))
        if node.declared:
            _row("declared metadata", json.dumps(
                node.declared, ensure_ascii=False, sort_keys=True))
        _row("provenance", node.provenance)
    elif isinstance(node, RobotEmbodiment):
        _row("slug", node.slug)
        _row("contract digest", node.contract_digest)
        if node.joint_names:
            _row("ordered joints", ", ".join(node.joint_names))
        if node.control_dt_s is not None:
            _row("control dt s", str(node.control_dt_s))
        if node.observation_contract:
            _row("observation contract", json.dumps(
                node.observation_contract, ensure_ascii=False, sort_keys=True))
        if node.action_contract:
            _row("action contract", json.dumps(
                node.action_contract, ensure_ascii=False, sort_keys=True))
        _row("provenance", node.provenance)
    elif isinstance(node, SoftwareEnvironment):
        _row("lock digest", node.lock_digest)
        if node.code_commit:
            _row("code commit", node.code_commit)
        if node.code_dirty is not None:
            _row("code dirty", str(node.code_dirty).lower())
        if node.code_tree_digest:
            _row("code tree digest", node.code_tree_digest)
        if node.code_diff_digest:
            _row("code diff digest", node.code_diff_digest)
        if node.capture_schema is not None:
            _row("capture schema", str(node.capture_schema))
        if node.captured_source_sha256:
            _row("captured source sha256", node.captured_source_sha256)
        if node.versions:
            _row("versions", json.dumps(
                node.versions, ensure_ascii=False, sort_keys=True))
        if node.runtime:
            _row("runtime", json.dumps(
                node.runtime, ensure_ascii=False, sort_keys=True))
        _row("provenance", node.provenance)
    elif isinstance(node, TrainingRun):
        _row("project", node.project)
        _row("run id", node.run_id)
        _row("requested initialization", node.requested_initialization_mode)
        _row("observed initialization",
             node.observed_initialization_mode or "not observed")
        if node.selection_digest:
            _row("selection digest", node.selection_digest)
        if node.code_commit:
            _row("code commit", node.code_commit)
        _row("created at", str(node.created_at))
        _row("provenance", node.provenance)
    elif isinstance(node, TrainingIteration):
        _row("project", node.project)
        _row("run id", node.run_id)
        _row("iteration", str(node.iteration_index))
        _row("provenance", node.provenance)

    if active_reason:
        rows.append(
            f"<div style='margin-top:8px; padding:4px 6px; "
            f"background:#3a2e00; border-radius:4px;'>"
            f"<b style='color:#ffd54f;'>ACTIVE:</b> "
            f"{html.escape(active_reason)}</div>")

    body = "".join(rows) or "<i>(no extra fields)</i>"
    return f"<div style='font-family:sans-serif; max-width:360px'>{header}{body}</div>"


# ── Provenance lookup ───────────────────────────────────────────────────
def _active_sets(provenance: dict) -> tuple[set[str], set[str]]:
    """Return (active_target_terms, active_arxiv_ids) from a provenance
    dict. Only entries with `still_active=True` count."""
    active_terms: set[str] = set()
    active_arxiv_ids: set[str] = set()
    for term, entries in (provenance or {}).items():
        for entry in entries or []:
            if entry.get("still_active"):
                active_terms.add(term)
                aid = entry.get("arxiv_id")
                if aid:
                    active_arxiv_ids.add(str(aid))
    return active_terms, active_arxiv_ids


def _active_reason_for(
    node: Any, *, active_terms: set[str], active_arxiv_ids: set[str],
) -> str | None:
    if isinstance(node, Paper) and node.arxiv_id in active_arxiv_ids:
        return "cited by active edits in provenance.json"
    if isinstance(node, (Technique, FailureMode, RewardComponent, Environment)):
        name = getattr(node, "name", "")
        if name in active_terms:
            return "target_term appears in provenance.json with still_active=True"
    return None


# ── Main entry ──────────────────────────────────────────────────────────
def build_kg_html(
    store: SculptorKG,
    out_path: Path | str,
    *,
    provenance: dict | None = None,
    title: str = "Reward Sculptor — Knowledge Graph",
) -> VizResult:
    """Render the KG to a self-contained HTML via pyvis."""
    from pyvis.network import Network

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    active_terms, active_arxiv_ids = _active_sets(provenance or {})

    # Pyvis dark theme, wide canvas, physics layout.
    net = Network(
        height="100vh", width="100%",
        bgcolor="#0b0f1a", font_color="#e8e8ed",
        directed=True, notebook=False, cdn_resources="in_line",
    )
    # §2026-07-03 smoothness pass — options ADAPT to graph size. The old
    # fixed options were tuned on a ~100-node graph; on the unified shared
    # graph (~1.5k nodes / ~1.6k edges) per-edge labels + bezier smoothing
    # + always-on physics dropped interaction to a slideshow. Large graphs:
    # straight edges, no edge labels (relation stays in the hover tooltip),
    # forceAtlas2 (better cluster separation at scale), and physics is
    # FROZEN after stabilization (see _CONTROLS_JS) so pan/zoom stays
    # butter-smooth; a toolbar button re-enables it on demand.
    n_edges_total = sum(1 for _ in store.all_edges())
    big = n_edges_total > 600
    smooth = "false" if big else (
        '{"enabled": true, "type": "continuous"}')
    solver_block = (
        '"solver": "forceAtlas2Based", "forceAtlas2Based": {'
        '"gravitationalConstant": -85, "centralGravity": 0.012, '
        '"springLength": 120, "springConstant": 0.06, "damping": 0.4, '
        '"avoidOverlap": 0.3}'
        if big else
        '"solver": "barnesHut", "barnesHut": {'
        '"gravitationalConstant": -20000, "centralGravity": 0.35, '
        '"springLength": 160, "springConstant": 0.04, "damping": 0.12, '
        '"avoidOverlap": 0.6}'
    )
    stabilization_iters = 200 if big else 250
    # vis.js's improvedLayout (Kamada-Kawai initial placement) is
    # quadratic in node count — on the ~1.5k-node shared graph it FROZE
    # the tab for tens of seconds before physics even started.
    improved_layout = "false" if big else "true"
    net.set_options(f"""
    {{
      "layout": {{"improvedLayout": {improved_layout}}},
      "physics": {{
        "enabled": true,
        {solver_block},
        "minVelocity": 0.75,
        "stabilization": {{"iterations": {stabilization_iters},
                           "updateInterval": 50}}
      }},
      "nodes": {{
        "font": {{"size": 16, "color": "#e8e8ed", "face": "arial"}},
        "borderWidth": 2,
        "shape": "dot"
      }},
      "edges": {{
        "smooth": {smooth},
        "arrows": {{"to": {{"enabled": true, "scaleFactor": 0.6}}}},
        "font": {{"size": 11, "color": "#9fb3c8", "strokeWidth": 0,
                  "align": "middle"}}
      }},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": true,
        "keyboard": {{"enabled": true, "bindToWindow": false}},
        "navigationButtons": false
      }}
    }}
    """)

    # Add nodes. Resolve all at once so we can size `Paper` by outdegree +
    # highlight actives.
    all_nodes = store.find_nodes()
    node_degree: dict[str, int] = {}
    edges_cache: list[Any] = list(store.all_edges())
    for edge in edges_cache:
        node_degree[edge.src] = node_degree.get(edge.src, 0) + 1
        node_degree[edge.dst] = node_degree.get(edge.dst, 0) + 1

    n_active = 0
    added_ids: set[str] = set()   # track what actually made it into the network
    for node in all_nodes:
        kind = type(node).__name__
        palette = _NODE_COLORS.get(kind, _FALLBACK_COLOR)
        reason = _active_reason_for(
            node, active_terms=active_terms, active_arxiv_ids=active_arxiv_ids)
        is_active = reason is not None
        if is_active:
            n_active += 1
            color = {
                "background": _ACTIVE_COLOR["background"],
                "border": _ACTIVE_COLOR["border"],
                "highlight": {
                    "background": _ACTIVE_COLOR["background"],
                    "border": "#ffffff",
                },
            }
            size = 34
            border_width = 4
        else:
            border = palette["border"]
            border_width = 2
            # RunCase: verdict on the border — green helped / red
            # regressed — so the experience silo reads at a glance.
            if isinstance(node, RunCase):
                v = _VERDICT_BORDER.get(node.verdict)
                if v:
                    border = v
                    border_width = 3
            color = {
                "background": palette["background"],
                "border": border,
                "highlight": {
                    "background": palette["background"],
                    "border": "#ffffff",
                },
            }
            # Scale by degree so hub papers pop out slightly.
            size = 16 + min(node_degree.get(node.id, 0), 8) * 2

        net.add_node(
            node.id, label=_label_for(node),
            title=_tooltip_for(node, reason),
            color=color, size=size, borderWidth=border_width,
            # Diamonds separate "this system's own experience" from the
            # published-literature dots without needing the legend.
            shape=_NODE_SHAPES.get(kind, "dot"),
        )
        added_ids.add(node.id)

    # Add edges. Label with relation; color by relation. SKIP an edge whose
    # endpoint is not in the node set — a DANGLING edge (a ref to a node that was
    # never persisted, e.g. an unhealed `failure:…` stub) would otherwise make
    # pyvis raise `AssertionError: non existent node …` and 500 the whole viz. A
    # visualization must degrade gracefully on a slightly-inconsistent graph, so we
    # drop the un-drawable edge and record the count (run `kg/heal-stubs` to repair
    # the underlying dangling refs).
    n_dangling = 0
    node_index = {n.id: n for n in all_nodes}
    for edge in edges_cache:
        if edge.src not in added_ids or edge.dst not in added_ids:
            n_dangling += 1
            continue
        rel = edge.relation.value if hasattr(edge.relation, "value") else str(edge.relation)
        edge_kwargs: dict[str, Any] = {
            "title": _edge_tooltip(edge),
            "color": _EDGE_COLORS.get(rel, _DEFAULT_EDGE_COLOR),
            "width": 2 if _edge_touches_active(
                edge, active_terms, active_arxiv_ids, node_index) else 1,
        }
        # Per-edge text labels are the single biggest FPS cost at scale;
        # on big graphs the relation lives in the hover tooltip instead.
        if not big:
            edge_kwargs["label"] = rel.replace("_", " ").title()
        net.add_edge(edge.src, edge.dst, **edge_kwargs)

    # pyvis's write_html has flaky template paths on some machines; use
    # generate_html + manual write.
    body_html = net.generate_html(notebook=False)
    # Patch in the page title + the control panel (legend, search, kind
    # filters, physics toggle).
    kind_counts: dict[str, int] = {}
    for node in all_nodes:
        kind_counts[type(node).__name__] = (
            kind_counts.get(type(node).__name__, 0) + 1)
    node_kinds = {n.id: type(n).__name__ for n in all_nodes
                  if n.id in added_ids}
    body_html = _inject_title_and_legend(
        body_html, title=title, n_active=n_active, kind_counts=kind_counts)
    body_html = _inject_controls(body_html, node_kinds=node_kinds)
    # Phase 7e: postMessage click events so the UI's GraphModal can
    # open a side-pane paper detail when a node is clicked.
    body_html = _inject_click_forwarder(body_html)
    out_path.write_text(body_html, encoding="utf-8")

    return VizResult(
        out_path=out_path,
        n_nodes=len(all_nodes),
        n_edges=len(edges_cache) - n_dangling,   # edges actually drawn
        n_active_nodes=n_active,
    )


def _edge_tooltip(edge) -> str:
    rel = edge.relation.value if hasattr(edge.relation, "value") else str(edge.relation)
    parts = [f"<b>{rel}</b>"]
    if edge.data:
        # Edge data is the requested/observed/authority receipt for artifact
        # lineage. Rendering only ``evidence`` silently hid the exact hashes,
        # load roles, world pins, and worker observations that make these
        # relations scientifically meaningful. Stored edge data is JSON, so
        # it is safe to render deterministically in full after HTML escaping.
        receipt = json.dumps(
            edge.data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        parts.append(
            "<div style='margin-top:6px;color:#8fb5ff'>structured receipt</div>"
            "<pre style='margin:4px 0 0;max-width:520px;max-height:300px;"
            "overflow:auto;white-space:pre-wrap;word-break:break-word;"
            "font-size:11px;line-height:1.35'>"
            f"{html.escape(receipt)}"
            "</pre>"
        )
    else:
        parts.append(
            "<div style='margin-top:4px;color:#ffca80'>"
            "no structured edge receipt recorded</div>"
        )
    return (
        "<div style='font-family:sans-serif;max-width:540px'>"
        + "".join(parts)
        + "</div>"
    )


def _edge_touches_active(
    edge, active_terms: set[str], active_arxiv_ids: set[str],
    node_index: dict[str, Any],
) -> bool:
    for node_id in (edge.src, edge.dst):
        node = node_index.get(node_id)
        if node is None:
            continue
        if isinstance(node, Paper) and node.arxiv_id in active_arxiv_ids:
            return True
        if hasattr(node, "name") and node.name in active_terms:
            return True
    return False


_KIND_LEGEND_ORDER = (
    ("Paper", "#4fc3f7", "dot"),
    ("Technique", "#81c784", "dot"),
    ("FailureMode", "#ef5350", "dot"),
    ("RewardComponent", "#ba68c8", "dot"),
    ("Environment", "#ffb74d", "dot"),
    ("Result", "#90a4ae", "dot"),
    ("RunCase", "#4dd0e1", "diamond"),
    ("ResearchCapability", "#64b5f6", "dot"),
    ("ImplementationStatus", "#fff176", "dot"),
    ("PolicyArtifact", "#26a69a", "box"),
    ("ReferenceMotion", "#ab47bc", "box"),
    ("WorldArtifact", "#ffa726", "box"),
    ("ModeExecutionArtifact", "#ec407a", "box"),
    ("ArtifactAttestation", "#ffee58", "hexagon"),
    ("RobotEmbodiment", "#66bb6a", "ellipse"),
    ("SoftwareEnvironment", "#78909c", "database"),
    ("TrainingRun", "#42a5f5", "triangle"),
    ("TrainingIteration", "#26c6da", "dot"),
)

_KIND_LABELS = {
    "RunCase": "Run experience",
    "PolicyArtifact": "Policy artifact",
    "ReferenceMotion": "Reference motion",
    "WorldArtifact": "World artifact",
    "ModeExecutionArtifact": "Mode execution",
    "ArtifactAttestation": "Artifact attestation",
    "RobotEmbodiment": "Robot embodiment",
    "SoftwareEnvironment": "Software environment",
    "TrainingRun": "Training run",
    "TrainingIteration": "Training iteration",
}

_LEGEND_SHAPE_CLASSES = {
    "diamond": "kg-diamond",
    "box": "kg-box",
    "hexagon": "kg-hexagon",
    "ellipse": "kg-ellipse",
    "database": "kg-database",
    "triangle": "kg-triangle",
}

_LEGEND_HTML = """
<style>
  #sculpt-legend input[type=checkbox] { accent-color: #4fc3f7; }
  #sculpt-legend label { cursor: pointer; user-select: none; }
  #sculpt-legend label:hover { color: #ffffff; }
  #kg-search { width: 100%; box-sizing: border-box; background: #101827;
    color: #e8e8ed; border: 1px solid #2a3444; border-radius: 6px;
    padding: 6px 8px; font-size: 13px; outline: none; }
  #kg-search:focus { border-color: #4fc3f7; }
  #kg-physics-btn { width: 100%; margin-top: 8px; background: #16233a;
    color: #cfe3ff; border: 1px solid #2a3444; border-radius: 6px;
    padding: 6px 8px; font-size: 12px; cursor: pointer; }
  #kg-physics-btn:hover { background: #1d2f4e; }
  #kg-search-status { font-size: 11px; opacity: 0.7; min-height: 14px;
    margin-top: 3px; }
  .kg-diamond { transform: rotate(45deg); border-radius: 2px !important; }
  .kg-box { border-radius: 2px !important; }
  .kg-hexagon { border-radius: 2px !important;
    clip-path: polygon(25% 5%,75% 5%,100% 50%,75% 95%,25% 95%,0 50%); }
  .kg-ellipse { width:16px !important; height:11px !important; }
  .kg-database { border-radius: 50% 50% 3px 3px !important; }
  .kg-triangle { border-radius: 0 !important;
    clip-path: polygon(50% 0,100% 100%,0 100%); }
</style>
<div id="sculpt-legend" style="
    position: fixed; top: 16px; left: 16px; z-index: 1000;
    background: rgba(10, 15, 25, 0.94); color: #e8e8ed;
    padding: 14px 16px; border-radius: 10px;
    border: 1px solid #2a3444;
    font-family: Segoe UI, Arial, sans-serif; font-size: 13px;
    max-width: 300px; max-height: calc(100vh - 32px); overflow-y: auto;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45);">
  <div style="font-size: 15px; font-weight: 600; margin-bottom: 8px;">
    __TITLE__
  </div>
  <input id="kg-search" type="search"
         placeholder="Search nodes… (Enter = next match)" />
  <div id="kg-search-status"></div>
  <div style="opacity: 0.75; margin: 8px 0 10px; font-size: 12px;">
    __ACTIVE_NOTE__
  </div>
  <div style="display: grid; grid-template-columns: 16px auto;
              row-gap: 5px; column-gap: 8px; align-items: center;">
    <span style="width:14px; height:14px; background:#ffd54f; border-radius:50%; border:1px solid #ffa000;"></span><span>Active (provenance)</span>
__KIND_ROWS__
  </div>
  <div style="opacity:0.72; margin-top:9px; font-size:11px; line-height:1.35;">
    Boxes are immutable artifacts; hexagons are admission attestations;
    triangles are concrete training runs. Hover an arrow for its full
    structured requested/observed lineage receipt.
  </div>
  <button id="kg-physics-btn" title="Re-run the force layout">
    ↻ re-run layout
  </button>
</div>
"""

_KIND_ROW_TEMPLATE = (
    '    <span class="{shape_cls}" style="width:14px; height:14px; '
    'background:{color}; border-radius:50%;"></span>'
    '<label><input type="checkbox" class="kg-kind-toggle" '
    'data-kind="{kind}" checked style="margin-right:6px; '
    'vertical-align:middle;">{kind_label} ({count})</label>'
)


def _inject_title_and_legend(html_src: str, *, title: str, n_active: int,
                             kind_counts: dict[str, int]) -> str:
    rows = []
    for kind, color, shape in _KIND_LEGEND_ORDER:
        count = kind_counts.get(kind, 0)
        if count == 0:
            continue
        label = _KIND_LABELS.get(kind, kind)
        rows.append(_KIND_ROW_TEMPLATE.format(
            kind=kind, kind_label=label, color=color, count=count,
            shape_cls=_LEGEND_SHAPE_CLASSES.get(shape, "")))
    active_note = (
        f"{n_active} node(s) in gold — cited by the current project's "
        f"provenance (still_active)."
        if n_active else
        "Diamonds are this system's OWN run experience (border: green "
        "helped / red regressed). Uncheck kinds to filter."
    )
    legend = (_LEGEND_HTML
              .replace("__TITLE__", html.escape(title))
              .replace("__ACTIVE_NOTE__", html.escape(active_note))
              .replace("__KIND_ROWS__", "\n".join(rows)))
    # Set <title>
    html_src = html_src.replace(
        "<title>",
        f"<title>{html.escape(title)} – ",
        1,
    )
    # Insert legend right after <body> so it floats above the network.
    return html_src.replace("<body>", "<body>\n" + legend, 1)


# §2026-07-03 interactivity: physics freeze-after-stabilize (smooth pan/
# zoom on big graphs), search with focus/cycling, per-kind visibility
# filters. Injected before </body>; polls for the pyvis `network` global
# the same way the click forwarder does.
_CONTROLS_JS_TEMPLATE = """
<script>
window.__KG_KINDS__ = __KINDS_JSON__;
(function () {
  function ready() {
    if (typeof network === 'undefined' || !network ||
        typeof nodes === 'undefined' || !nodes) {
      setTimeout(ready, 100);
      return;
    }
    // ── freeze physics once stabilized: interaction stays smooth and the
    // layout stops drifting under the cursor.
    network.once('stabilizationIterationsDone', function () {
      network.setOptions({ physics: false });
      network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
    });
    var physicsBtn = document.getElementById('kg-physics-btn');
    if (physicsBtn) {
      physicsBtn.addEventListener('click', function () {
        network.setOptions({ physics: true });
        network.stabilize(300);
        network.once('stabilizationIterationsDone', function () {
          network.setOptions({ physics: false });
        });
      });
    }

    // ── per-kind visibility filters.
    var kinds = window.__KG_KINDS__ || {};
    document.querySelectorAll('.kg-kind-toggle').forEach(function (box) {
      box.addEventListener('change', function () {
        var kind = box.getAttribute('data-kind');
        var hidden = !box.checked;
        var updates = [];
        Object.keys(kinds).forEach(function (id) {
          if (kinds[id] === kind) updates.push({ id: id, hidden: hidden });
        });
        if (updates.length) nodes.update(updates);
      });
    });

    // ── search: substring over labels; Enter cycles matches; focuses +
    // selects each hit with a smooth animated pan.
    var input = document.getElementById('kg-search');
    var status = document.getElementById('kg-search-status');
    var matches = [];
    var cursor = -1;
    function runSearch(q) {
      matches = [];
      cursor = -1;
      if (!q) { if (status) status.textContent = ''; return; }
      q = q.toLowerCase();
      nodes.forEach(function (n) {
        var hay = String(n.label || '') + ' ' + String(n.id || '');
        if (hay.toLowerCase().indexOf(q) !== -1 && !n.hidden) {
          matches.push(n.id);
        }
      });
      if (status) {
        status.textContent = matches.length
          ? matches.length + ' match(es) — Enter to cycle'
          : 'no matches';
      }
      if (matches.length) focusNext();
    }
    function focusNext() {
      if (!matches.length) return;
      cursor = (cursor + 1) % matches.length;
      var id = matches[cursor];
      network.selectNodes([id]);
      network.focus(id, {
        scale: 1.1,
        animation: { duration: 500, easingFunction: 'easeInOutQuad' }
      });
      if (status) {
        status.textContent =
          (cursor + 1) + '/' + matches.length + ' — Enter for next';
      }
    }
    if (input) {
      var deb = null;
      input.addEventListener('input', function () {
        clearTimeout(deb);
        deb = setTimeout(function () { runSearch(input.value.trim()); }, 200);
      });
      input.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') { ev.preventDefault(); focusNext(); }
      });
    }
  }
  ready();
})();
</script>
"""


def _inject_controls(html_src: str, *, node_kinds: dict[str, str]) -> str:
    script = _CONTROLS_JS_TEMPLATE.replace(
        "__KINDS_JSON__", json.dumps(node_kinds))
    if "</body>" in html_src:
        return html_src.replace("</body>", script + "</body>", 1)
    return html_src + script


# Click-forwarding shim. Appended right before `</body>` so it runs
# after pyvis has created the `network` global. Posts a typed message
# to the parent window on every node click — the GraphModal listens
# and opens a side-pane paper detail. Cheap + self-contained (no new
# deps); falls back silently when not inside an iframe.
_CLICK_FORWARDER_JS = """
<script>
(function () {
  function hook() {
    if (typeof network === 'undefined' || !network || !network.on) {
      setTimeout(hook, 100);
      return;
    }
    network.on('click', function (params) {
      if (!params || !params.nodes || params.nodes.length === 0) return;
      var nodeId = String(params.nodes[0]);
      var kind = '';
      var arxivId = null;
      if (nodeId.indexOf('paper:') === 0) {
        kind = 'Paper';
        arxivId = nodeId.slice('paper:'.length);
      } else if (nodeId.indexOf('technique:') === 0) {
        kind = 'Technique';
      } else if (nodeId.indexOf('failure_mode:') === 0) {
        kind = 'FailureMode';
      } else if (nodeId.indexOf('reward_component:') === 0) {
        kind = 'RewardComponent';
      } else if (nodeId.indexOf('environment:') === 0) {
        kind = 'Environment';
      } else if (nodeId.indexOf('case:') === 0) {
        kind = 'RunCase';
      }
      try {
        window.parent.postMessage({
          type: 'kg_node_click',
          id: nodeId,
          kind: kind,
          arxiv_id: arxivId,
        }, '*');
      } catch (e) {
        // parent may be cross-origin; no-op.
      }
    });
  }
  hook();
})();
</script>
"""


def _inject_click_forwarder(html_src: str) -> str:
    """Append the click-forwarding <script> before </body> so it runs
    after pyvis has created the `network` global."""
    if "</body>" in html_src:
        return html_src.replace("</body>", _CLICK_FORWARDER_JS + "</body>", 1)
    return html_src + _CLICK_FORWARDER_JS
