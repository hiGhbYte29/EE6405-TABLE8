"""Turn retrieved chunks into (head, relation, tail) triples.

Two extractors, deliberately -- the report can compare them:

``extract_triples_cooccurrence``  statistical, no LLM, no key, reproducible;
                                  the honest baseline for the KG section
``extract_triples_llm``           prompts the LLM for typed relations; richer
                                  labels, but needs validation against the
                                  co-occurrence graph
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from ee6405.agent.llm import LLM, build_llm
from ee6405.types import Chunk

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#-]{1,}")

# Terms worth keeping as graph nodes even though they are short / lowercase.
SEED_TERMS = {
    "tf-idf", "lda", "word2vec", "glove", "fasttext", "elmo", "bert", "roberta",
    "albert", "distilbert", "gpt", "lstm", "gru", "crf", "svm", "gcn", "gat",
    "gnn", "rag", "transformer", "attention", "embedding", "tokenization",
    "ner", "pos", "parsing", "coreference", "chunking", "summarization",
    "sentiment", "bleu", "rouge", "meteor", "perplexity", "n-gram",
}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "can", "will", "which", "when", "what", "how", "its", "has", "have", "not",
    "but", "you", "your", "their", "there", "than", "then", "also", "each",
    "page", "slide", "week", "lecture", "figure", "table", "example",
}


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str
    weight: float = 1.0
    source: tuple[str, ...] = field(default_factory=tuple)


def _format_chunk_source(chunk: Chunk) -> str:
    meta = getattr(chunk, "metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}

    doc_title = meta.get("doc_title") or getattr(chunk, "doc_title", None)
    if not doc_title:
        src_path = meta.get("source") or getattr(chunk, "source", None)
        if src_path:
            doc_title = Path(src_path).stem
        else:
            week = meta.get("week") or getattr(chunk, "week", "")
            doc_title = f"{week}_Material" if week else "CourseDoc"
	
    locator = meta.get("locator") or getattr(chunk, "locator", None)
    page = meta.get("page") or getattr(chunk, "page", None)

    if locator:
        loc_str = str(locator)
    elif page is not None:
        loc_str = f"page:{page}"
    else:
        loc_str = "N/A"

    return f"{doc_title} [{loc_str}]"


def _terms(text: str, min_len: int = 4) -> list[str]:
    found = []
    lowered = text.lower()
    for seed in SEED_TERMS:
        if seed in lowered:
            found.append(seed)
    for word in _WORD.findall(text):
        w = word.lower()
        if len(w) >= min_len and w not in STOPWORDS and not w.isdigit():
            found.append(w)
    return found


def extract_triples_cooccurrence(
    chunks: list[Chunk], top_terms: int = 60, min_count: int = 2
) -> list[Triple]:
    """Link terms that repeatedly appear in the same chunk.

    Relation is always ``co_occurs_with``; ``weight`` is the co-occurrence count.
    """
    freq = Counter()
    per_chunk: list[tuple[set[str], str]] = []
    for chunk in chunks:
        terms = set(_terms(chunk.text))
        src_tag = _format_chunk_source(chunk)
        per_chunk.append((terms, src_tag))
        freq.update(terms)

    keep = {term for term, n in freq.most_common(top_terms) if n >= min_count}
    
    pairs_count = Counter()
    pairs_source = defaultdict(set)
    for terms, src_tag in per_chunk:
        for a, b in combinations(sorted(terms & keep), 2):
            pair = (a, b)
            pairs_count[pair] += 1
            if src_tag:
                pairs_source[pair].add(src_tag)

    return [
        Triple(head=a, relation="co_occurs_with", tail=b, weight=float(n), source=tuple(sorted(pairs_source[(a, b)])),)
        for (a, b), n in pairs_count.items()
        if n >= min_count
    ]


_LLM_PROMPT = """Extract knowledge-graph triples from the NLP course text below.

Return ONLY a JSON array, no prose. Each element:
  {"head": "...", "relation": "...", "tail": "..."}

Use short noun phrases for head/tail and a lowercase snake_case verb phrase for
relation (e.g. "is_a", "uses", "improves_on", "evaluated_by", "part_of").
Extract at most 12 triples. Skip anything not stated in the text.

TEXT:
"""


def extract_triples_llm(chunks: list[Chunk], llm: LLM | None = None) -> list[Triple]:
    """Ask the LLM for typed relations, one call per chunk.

    Malformed JSON from a chunk is skipped rather than raised -- a long run
    must not die on one bad response.
    """
    llm = llm or build_llm()
    triples: list[Triple] = []
    for chunk in chunks:
        src_tag = _format_chunk_source(chunk)
        raw = llm.chat([{"role": "user", "content": _LLM_PROMPT + chunk.text}])
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            continue
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        for item in items:
            if isinstance(item, dict) and {"head", "relation", "tail"} <= item.keys():
                triples.append(
                    Triple(
                        head=str(item["head"]).strip().lower(),
                        relation=str(item["relation"]).strip().lower(),
                        tail=str(item["tail"]).strip().lower(),
                        source=(src_tag,) if src_tag else (),
                    )
                )
    return triples