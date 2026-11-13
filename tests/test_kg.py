"""
triples = extract_triples_cooccurrence(chunks)
g = build_graph(triples)
html = to_pyvis_html(g)

output_file = "test_graph.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"成功生成知识图谱网页：{output_file}")
print(f"图谱节点数（概念数）: {g.number_of_nodes()}，边数（关系数）: {g.number_of_edges()}")
"""

from ee6405.kg.extract import extract_triples_cooccurrence
from ee6405.kg.graph import build_graph, to_pyvis_html
from ee6405.rag.store import ChunkStore

store = ChunkStore("data/processed/index.sqlite")
query = "transformer"
hits = store.search_bm25(query, top_k=10)
chunks = [h[0] for h in hits]

triples = extract_triples_cooccurrence(chunks, top_terms=40, min_count=2)
full_g = build_graph(triples)

if full_g.has_node(query):
    neighbors = [
        (neighbor, full_g[query][neighbor].get("weight", 1))
        for neighbor in full_g.neighbors(query)
    ]
    top_neighbors = [n for n, _ in sorted(neighbors, key=lambda x: -x[1])[:10]]
    
    sub_nodes = set([query] + top_neighbors)
    display_g = full_g.subgraph(sub_nodes).copy()
else:
    top_10_nodes = [node for node, _ in sorted(full_g.degree, key=lambda x: -x[1])[:10]]
    display_g = full_g.subgraph(top_10_nodes).copy()

html = to_pyvis_html(display_g)
with open("test_graph.html", "w", encoding="utf-8") as f:
    f.write(html)