"""P3 -- knowledge graph extraction and visualisation over the course corpus."""

from ee6405.kg.extract import Triple, extract_triples_cooccurrence, extract_triples_llm
from ee6405.kg.course_map import (
    CourseMap,
    build_course_map,
    build_query_map,
    scan_corpus,
    triples_to_map,
)
from ee6405.kg.graph import build_graph, to_agraph, to_pyvis_html
from ee6405.kg.render import to_obsidian_html

__all__ = [
    "Triple",
    "extract_triples_cooccurrence",
    "extract_triples_llm",
    "build_graph",
    "to_pyvis_html",
    "to_agraph",
    "CourseMap",
    "scan_corpus",
    "build_course_map",
    "build_query_map",
    "triples_to_map",
    "to_obsidian_html",
]
