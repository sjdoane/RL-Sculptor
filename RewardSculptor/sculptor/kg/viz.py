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
    Environment,
    FailureMode,
    Paper,
    RewardComponent,
    Result,
    Technique,
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
}
_ACTIVE_COLOR = {"background": "#ffd54f", "border": "#ffa000"}  # gold halo
_FALLBACK_COLOR = {"background": "#bdbdbd", "border": "#757575"}

_EDGE_COLORS = {
    "INTRODUCES":    "#66bb6a",
    "ADDRESSES":     "#ef5350",
    "USES":          "#ba68c8",
    "EVALUATES_ON":  "#ffb74d",
    "CITES":         "#78909c",
    "REPORTS":       "#9ccc65",
    "IMPROVES_OVER": "#42a5f5",
}
_DEFAULT_EDGE_COLOR = "#b0bec5"


@dataclass
class VizResult:
    out_path: Path
    n_nodes: int
    n_edges: int
    n_active_nodes: int


# ── Tooltip / label helpers ─────────────────────────────────────────────
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
        return f"cited by active edits in provenance.json"
    if isinstance(node, (Technique, FailureMode, RewardComponent, Environment)):
        name = getattr(node, "name", "")
        if name in active_terms:
            return f"target_term appears in provenance.json with still_active=True"
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
        height="820px", width="100%",
        bgcolor="#0b0f1a", font_color="#e8e8ed",
        directed=True, notebook=False, cdn_resources="in_line",
    )
    # Barnes-Hut physics with gentler gravity for legible clusters.
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "solver": "barnesHut",
        "barnesHut": {
          "gravitationalConstant": -20000,
          "centralGravity": 0.35,
          "springLength": 160,
          "springConstant": 0.04,
          "damping": 0.12,
          "avoidOverlap": 0.6
        },
        "minVelocity": 0.75,
        "stabilization": {"iterations": 250}
      },
      "nodes": {
        "font": {"size": 16, "color": "#e8e8ed", "face": "arial"},
        "borderWidth": 2,
        "shape": "dot"
      },
      "edges": {
        "smooth": {"enabled": true, "type": "continuous"},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
        "font": {"size": 11, "color": "#9fb3c8", "strokeWidth": 0, "align": "middle"}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": true
      }
    }
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
            color = {
                "background": palette["background"],
                "border": palette["border"],
                "highlight": {
                    "background": palette["background"],
                    "border": "#ffffff",
                },
            }
            # Scale by degree so hub papers pop out slightly.
            size = 16 + min(node_degree.get(node.id, 0), 8) * 2
            border_width = 2

        net.add_node(
            node.id, label=_label_for(node),
            title=_tooltip_for(node, reason),
            color=color, size=size, borderWidth=border_width,
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
        net.add_edge(
            edge.src, edge.dst,
            title=_edge_tooltip(edge),
            label=rel.replace("_", " ").title(),
            color=_EDGE_COLORS.get(rel, _DEFAULT_EDGE_COLOR),
            width=2 if _edge_touches_active(
                edge, active_terms, active_arxiv_ids, node_index) else 1,
        )

    # pyvis's write_html has flaky template paths on some machines; use
    # generate_html + manual write.
    body_html = net.generate_html(notebook=False)
    # Patch in the page title + legend.
    body_html = _inject_title_and_legend(body_html, title=title,
                                         n_active=n_active)
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
        ev = edge.data.get("evidence") if isinstance(edge.data, dict) else None
        if ev:
            parts.append(f"<div style='margin-top:4px'>{html.escape(str(ev))[:400]}</div>")
        src_paper = edge.data.get("source_paper_id") if isinstance(edge.data, dict) else None
        if src_paper:
            parts.append(f"<div style='margin-top:4px;color:#8fb5ff'>"
                         f"source_paper_id: {html.escape(str(src_paper))}</div>")
    return "<div style='font-family:sans-serif;max-width:360px'>" + "".join(parts) + "</div>"


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


_LEGEND_HTML = """
<div id="sculpt-legend" style="
    position: fixed; top: 16px; left: 16px; z-index: 1000;
    background: rgba(10, 15, 25, 0.94); color: #e8e8ed;
    padding: 14px 18px; border-radius: 10px;
    border: 1px solid #2a3444;
    font-family: Segoe UI, Arial, sans-serif; font-size: 13px;
    max-width: 330px; box-shadow: 0 4px 16px rgba(0,0,0,0.4);">
  <div style="font-size: 15px; font-weight: 600; margin-bottom: 8px;">
    __TITLE__
  </div>
  <div style="opacity: 0.75; margin-bottom: 10px; font-size: 12px;">
    __ACTIVE_COUNT__ node(s) highlighted in gold — cited by the current
    project's reports/provenance.json (still_active).
  </div>
  <div style="display: grid; grid-template-columns: 16px auto;
              row-gap: 4px; column-gap: 8px; align-items: center;">
    <span style="width:14px; height:14px; background:#ffd54f; border-radius:50%; border:1px solid #ffa000;"></span><span>Active (provenance)</span>
    <span style="width:14px; height:14px; background:#4fc3f7; border-radius:50%;"></span><span>Paper</span>
    <span style="width:14px; height:14px; background:#81c784; border-radius:50%;"></span><span>Technique</span>
    <span style="width:14px; height:14px; background:#ef5350; border-radius:50%;"></span><span>FailureMode</span>
    <span style="width:14px; height:14px; background:#ba68c8; border-radius:50%;"></span><span>RewardComponent</span>
    <span style="width:14px; height:14px; background:#ffb74d; border-radius:50%;"></span><span>Environment</span>
    <span style="width:14px; height:14px; background:#90a4ae; border-radius:50%;"></span><span>Result</span>
  </div>
</div>
"""


def _inject_title_and_legend(html_src: str, *,
                             title: str, n_active: int) -> str:
    legend = (_LEGEND_HTML
              .replace("__TITLE__", html.escape(title))
              .replace("__ACTIVE_COUNT__", str(n_active)))
    # Set <title>
    html_src = html_src.replace(
        "<title>",
        f"<title>{html.escape(title)} – ",
        1,
    )
    # Insert legend right after <body> so it floats above the network.
    return html_src.replace("<body>", "<body>\n" + legend, 1)


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
