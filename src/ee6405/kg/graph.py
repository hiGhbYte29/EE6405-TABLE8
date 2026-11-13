"""Build a networkx graph from triples and render it for the Streamlit app."""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

import networkx as nx

from ee6405.kg.extract import Triple

def build_graph(triples: list[Triple]) -> nx.Graph:
    """Undirected weighted graph. Parallel triples accumulate weight."""
    g = nx.Graph()
    for t in triples:
        for node in (t.head, t.tail):
            if not g.has_node(node):
                g.add_node(node, label=node, count=0, source=set())
            g.nodes[node]["count"] += 1
            if t.source:
                g.nodes[node]["source"].update(t.source)
        if g.has_edge(t.head, t.tail):
            g[t.head][t.tail]["weight"] += t.weight
            if t.source:
                g[t.head][t.tail]["source"].update(t.source)
        else:
            g.add_edge(t.head, t.tail, weight=t.weight, relation=t.relation, source=set(t.source))
    return g

def top_nodes(g: nx.Graph, n: int = 20) -> list[tuple[str, float]]:
    """Most central concepts -- the headline number for the KG slide."""
    if g.number_of_nodes() == 0:
        return []
    centrality = nx.degree_centrality(g)
    return sorted(centrality.items(), key=lambda kv: -kv[1])[:n]

def _get_node_color(val: float, min_v: float, max_v: float) -> dict:
    if max_v == min_v:
        ratio = 0.5
    else:
        ratio = (val - min_v) / (max_v - min_v)
    ratio = max(0.0, min(1.0, ratio))

    # Low:  rgb(186, 215, 248) -> #BAD7F8
    # High: rgb(37, 99, 235)   -> #2563EB
    r = int(186 - ratio * (186 - 37))
    g = int(215 - ratio * (215 - 99))
    b = int(248 - ratio * (248 - 235))
    bg_hex = f"#{r:02x}{g:02x}{b:02x}"

    return {
        "background": bg_hex,
        "border": "#1E40AF" if ratio > 0.6 else "#60A5FA",
        "highlight": {
            "background": "#8AD0D0",
            "border": "#6ECECE",
        },
        "hover": {
            "background": "#aed09c",
            "border": "#7f9f6f",
        },
    }

def to_pyvis_html(g: nx.Graph, height: str = "600px") -> str:
    """Interactive HTML for ``st.components.v1.html``."""
    from pyvis.network import Network

    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#222222")

    counts = [data.get("count", 1) for _, data in g.nodes(data=True)] or [1]
    min_c, max_c = min(counts), max(counts)

    for node, data in g.nodes(data=True):
        source_list = sorted(list(data.get("source", [])))
        if source_list:
            items_html = "".join(f"<li>{html.escape(str(s))}</li>" for s in source_list[:5])
            if len(source_list) > 5:
                items_html += f"<li class='kg-more'>... and {len(source_list) - 5} other sources</li>"
            sources_html = f"<ul class='kg-sources-list'>{items_html}</ul>"
        else:
            sources_html = "<div class='kg-empty'>No specific source</div>"

        cnt = max(1, int(data.get("count", 1)))
        node_color = _get_node_color(cnt, min_c, max_c)
        tooltip_html = (
            f"<div class='kg-tooltip'>"
            f"  <div class='kg-title'><b>Entity:</b> <span class='kg-name'>{html.escape(str(node))}</span></div>"
            f"  <div class='kg-item'><b>Count:</b> <span class='kg-badge'>{cnt}</span></div>"
            f"  <div class='kg-sources-header'><b>Source:</b></div>"
            f"  {sources_html}"
            f"</div>"
        )
        node_size = 14 + int(16 * ((cnt - min_c) / (max_c - min_c if max_c > min_c else 1)))

        net.add_node(
            node, label=data.get("label", node),
            value=cnt, size=node_size, title=tooltip_html,
            color=node_color, shape="dot", borderWidth=2,
            shadow={"enable": True, "size": 6, "x": 2, "y": 2},
            font={
                "size": 15,
                "face": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                "color": "#1e293b",
                "bold": {"mod": "bold"},
                "strokeWidth": 3,
                "strokeColor": "#ffffff",
            },
        )

    for u, v, data in g.edges(data=True):
        weight = float(data.get("weight", 1.0))
        edge_width = min(6.0, 1.5 + (weight * 0.6))
        rel = html.escape(str(data.get("relation", "related_to")))
        edge_tooltip = (
            f"<div class='kg-tooltip'>"
            f"  <div class='kg-title'><b>Relation:</b> <span class='kg-name'>{rel}</span></div>"
            f"  <div class='kg-item'><b>Count:</b> <span class='kg-badge'>{weight:g}</span></div>"
            f"</div>"
        )

        net.add_edge(
            u, v, value=weight,
            width=edge_width, title=edge_tooltip,
            color={"color": "rgba(148, 163, 184, 0.55)", "highlight": "#5AB8B8", "hover": "#7DA490"},
            smooth={"type": "continuous", "roundness": 0.2},
        )

    net.set_options(
        """
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -25000,
              "centralGravity": 0.05,
              "springLength": 240,
              "springConstant": 0.012,
              "damping": 0.15,
              "avoidOverlap": 0.8
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "hoverConnectedEdges": true,
            "selectConnectedEdges": true,
            "navigationButtons": false
          }
        }
        """
    )

    custom_css = """
    <style>
      div.vis-tooltip {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12) !important;
        padding: 10px 12px !important;
        white-space: normal !important;
        word-break: break-word !important;
        max-width: 320px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 13px !important;
        color: #334155 !important;
        line-height: 1.5 !important;
      }
      .kg-title { margin-bottom: 4px; font-size: 14px; color: #0f172a; }
      .kg-name { color: #2563eb; font-weight: 600; }
      .kg-item { margin-bottom: 4px; }
      .kg-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 9999px;
        padding: 0 8px;
        font-size: 12px;
        font-weight: 600;
      }
      .kg-sources-header { font-size: 12px; color: #64748b; margin-top: 6px; }
      .kg-sources-list { margin: 4px 0 0 16px; padding: 0; font-size: 12px; color: #475569; }
      .kg-sources-list li { margin-bottom: 2px; }
      .kg-more, .kg-empty { font-size: 12px; color: #94a3b8; font-style: italic; }
    </style>
    """

    custom_js = """
    <script>
      (function () {
        function htmlTitle(htmlStr) {
          var container = document.createElement("div");
          container.innerHTML = htmlStr;
          return container;
        }
        if (typeof nodes !== "undefined") {
          var updatedNodes = nodes.get()
            .filter(function (n) { return typeof n.title === "string"; })
            .map(function (n) { return { id: n.id, title: htmlTitle(n.title) }; });
          if (updatedNodes.length) nodes.update(updatedNodes);
        }
        if (typeof edges !== "undefined") {
          var updatedEdges = edges.get()
            .filter(function (e) { return typeof e.title === "string"; })
            .map(function (e) { return { id: e.id, title: htmlTitle(e.title) }; });
          if (updatedEdges.length) edges.update(updatedEdges);
        }
      })();
    </script>
    """

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.html"
        net.write_html(str(out), notebook=False)
        html_content = out.read_text(encoding="utf-8")
        html_content = html_content.replace("</head>", f"{custom_css}\n</head>", 1)
        html_content = html_content.replace("</body>", f"{custom_js}\n</body>", 1)
        return html_content

def to_agraph(g: nx.Graph, max_nodes: int = 120):
    """Nodes/edges for ``streamlit-agraph`` (lighter than pyvis in the app)."""
    from streamlit_agraph import Edge, Node

    keep = {n for n, _ in top_nodes(g, max_nodes)} or set(g.nodes)
    nodes = []
    for n in g.nodes:
        if n in keep:
            data = g.nodes[n]
            source_list = sorted(list(data.get("source", [])))
            source_preview = "\n".join(source_list[:4])
            tooltip = f"Entity: {n}\nCount: {data.get('count', 1)}\nSource:\n{source_preview or 'N/A'}"
            nodes.append(
                Node(
                    id=n,
                    label=str(n),
                    size=10 + 2 * min(int(data.get("count", 1)), 15),
                    title=tooltip,
                )
            )

    edges = [
        Edge(source=u, target=v, label=str(d.get("relation","")))
        for u, v, d in g.edge(data=True)
        if u in keep and v in keep
	]
	
    return nodes, edges