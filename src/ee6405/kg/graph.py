"""Build a networkx graph from triples and render it for the Streamlit app."""

from __future__ import annotations

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
                g.add_node(node, label=node, count=0)
            g.nodes[node]["count"] += 1
        if g.has_edge(t.head, t.tail):
            g[t.head][t.tail]["weight"] += t.weight
        else:
            g.add_edge(t.head, t.tail, weight=t.weight, relation=t.relation)
    return g


def top_nodes(g: nx.Graph, n: int = 20) -> list[tuple[str, float]]:
    """Most central concepts -- the headline number for the KG slide."""
    if g.number_of_nodes() == 0:
        return []
    centrality = nx.degree_centrality(g)
    return sorted(centrality.items(), key=lambda kv: -kv[1])[:n]


def to_pyvis_html(g: nx.Graph, height: str = "600px") -> str:
    """Interactive HTML for ``st.components.v1.html``."""
    from pyvis.network import Network

    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-8000, spring_length=120)
    for node, data in g.nodes(data=True):
        net.add_node(node, label=data.get("label", node),
                     value=max(1, int(data.get("count", 1))), title=node)
    for u, v, data in g.edges(data=True):
        net.add_edge(u, v, value=float(data.get("weight", 1.0)),
                     title=data.get("relation", ""))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.html"
        net.write_html(str(out), notebook=False)
        return out.read_text(encoding="utf-8")


def to_agraph(g: nx.Graph, max_nodes: int = 120):
    """Nodes/edges for ``streamlit-agraph`` (lighter than pyvis in the app)."""
    from streamlit_agraph import Edge, Node

    keep = {n for n, _ in top_nodes(g, max_nodes)} or set(g.nodes)
    nodes = [
        Node(id=n, label=str(n), size=10 + 2 * int(g.nodes[n].get("count", 1)))
        for n in g.nodes
        if n in keep
    ]
    edges = [
        Edge(source=u, target=v, label=str(d.get("relation", "")))
        for u, v, d in g.edges(data=True)
        if u in keep and v in keep
    ]
    return nodes, edges
