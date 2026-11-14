"""Course knowledge map: materials linked through the concepts they cover."""

from ee6405.kg.course_map import (
    build_course_map,
    build_query_map,
    concepts_in,
    material_label,
    scan_corpus,
    triples_to_map,
)
from ee6405.kg.extract import Triple
from ee6405.kg.render import to_obsidian_html
from ee6405.types import Chunk


def _chunk(i: int, source: str, text: str, modality: str = "slides", week: str = "W06") -> Chunk:
    return Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=text, source=source,
                 modality=modality, position=i, week=week)


CHUNKS = [
    _chunk(1, "raw/W06/W06_L02_LLMs.pdf", "BERT is pre-trained with masked language modeling."),
    _chunk(2, "raw/W06/W06_L02_LLMs.pdf", "RoBERTa drops NSP and changes how BERT is trained."),
    _chunk(3, "raw/W06/W06_Part2_Code.ipynb", "Load BERT with the tokenizer.", "notebook"),
    _chunk(4, "interim/W06_L02_Video.json", "So BERT and RoBERTa differ.", "transcript"),
    _chunk(5, "raw/W03/W03_TFIDF.pdf", "TF-IDF weights a term by rarity.", week="W03"),
]


def test_concepts_respect_word_boundaries():
    assert concepts_in("We compare RoBERTa and DistilBERT.") == {"RoBERTa", "DistilBERT"}
    assert "BERT" in concepts_in("bert-base-uncased")
    assert "Named Entity Recognition" in concepts_in("spaCy's NER tagger")
    assert concepts_in("the corner of the page") == set()


def test_material_label_is_readable():
    assert material_label(CHUNKS[3]) == "W06 Lecture 2 Video"
    assert material_label(CHUNKS[0]) == "W06 L02 LLMs"


def test_course_map_links_materials_through_shared_concepts():
    m = build_course_map(CHUNKS)
    groups = {n.id: n.group for n in m.nodes}
    assert groups["concept:BERT"] == "concept"
    assert groups["doc:raw/W06/W06_Part2_Code.ipynb"] == "code"
    bert_neighbours = {e.source for e in m.edges if e.target == "concept:BERT"}
    assert len(bert_neighbours) == 3  # slides, notebook and video all cover BERT
    # every edge joins a material to a concept, never two of the same kind
    assert all(groups[e.source] != "concept" and groups[e.target] == "concept" for e in m.edges)


def test_course_map_week_filter_and_shared_corpus():
    corpus = scan_corpus(CHUNKS)
    m = build_course_map(corpus, week="W03")
    assert [n.label for n in m.nodes if n.group != "concept"] == ["W03 TFIDF"]
    # building one map must not change what the next one sees
    assert len(build_course_map(corpus).nodes) == len(build_course_map(CHUNKS).nodes)


def test_query_map_marks_cited_materials():
    m = build_query_map("BERT vs RoBERTa?", CHUNKS[:2], CHUNKS)
    by_id = {n.id: n for n in m.nodes}
    assert by_id["query"].detail == "BERT vs RoBERTa?"
    assert by_id["doc:raw/W06/W06_L02_LLMs.pdf"].highlight
    assert {"concept:BERT", "concept:RoBERTa"} <= by_id.keys()
    assert any(e.source == "query" for e in m.edges)


def test_render_is_self_contained_and_escapes_script_end():
    m = triples_to_map([Triple("bert", "improves_on", "</script><b>elmo")])
    html = to_obsidian_html(m, height=300)
    assert "vis.Network" in html and "cdn" not in html.split("<script>__", 1)[0].lower()
    assert "</script><b>elmo" not in html
    assert "improves on" in html
