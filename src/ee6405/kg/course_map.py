"""Course knowledge map: course materials linked through the concepts they cover.

An Obsidian-style graph rather than a term co-occurrence graph. There are two
kinds of node -- a *material* (one slide deck, lecture video, notebook or quiz)
and a *concept* (TF-IDF, BERT, RAG ...) -- and an edge means "this material
covers this concept". Two materials are therefore related when they share
concepts, which is what makes the map useful for finding where else a topic is
taught.

No LLM and no embeddings: concepts come from a fixed vocabulary matched against
the indexed chunk text, so the map is reproducible and builds in about a second.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ee6405.types import Chunk

# (label shown on the map, regex alternatives). Matched case-insensitively with
# letter/digit boundaries, so "BERT" does not fire inside "RoBERTa".
CONCEPTS: list[tuple[str, str]] = [
    # --- text processing ---
    ("Tokenization", r"tokeni[sz](?:ation|er|ers|e|ed|ing)"),
    ("Stemming", r"stemm(?:ing|er|ers)"),
    ("Lemmatization", r"lemmati[sz](?:ation|er|e|ed|ing)"),
    ("Stop Words", r"stop[- ]?words?"),
    ("Regular Expressions", r"regular expressions?|regex(?:es)?"),
    ("N-grams", r"n[- ]?grams?|bigrams?|trigrams?|unigrams?"),
    ("Bag of Words", r"bag[- ]of[- ]words|bow"),
    ("TF-IDF", r"tf[- ]?idf"),
    ("One-hot Encoding", r"one[- ]hot"),
    ("Cosine Similarity", r"cosine[- _]similarit(?:y|ies)"),
    ("BM25", r"bm25"),
    ("PMI", r"pmi|pointwise mutual information"),
    # --- linguistic analysis ---
    ("POS Tagging", r"part[- ]of[- ]speech|pos[- _]tag(?:s|ging|ger)?"),
    ("Named Entity Recognition", r"named[- ]entit(?:y|ies)|ner"),
    ("Dependency Parsing", r"dependency pars(?:ing|er|e)|dependency tree"),
    ("Constituency Parsing", r"constituency pars(?:ing|er|e)|parse tree"),
    ("Coreference", r"coreference"),
    ("Information Extraction", r"information extraction|relation extraction"),
    ("WordNet", r"wordnet"),
    # --- embeddings ---
    ("Word Embeddings", r"(?:word )?embeddings?"),
    ("Word2Vec", r"word2vec|skip[- ]?gram|cbow"),
    ("GloVe", r"glove"),
    ("fastText", r"fasttext"),
    ("ELMo", r"elmo"),
    # --- topic models / classic ML ---
    ("Topic Modeling", r"topic model(?:s|ing|ling)?"),
    ("LDA", r"lda|latent dirichlet"),
    ("LSA", r"lsa|latent semantic"),
    ("Naive Bayes", r"na[iï]ve bayes"),
    ("Logistic Regression", r"logistic regression"),
    ("SVM", r"svms?|support vector"),
    ("Clustering", r"clustering|k[- ]means"),
    ("PCA", r"pca|principal component"),
    ("HMM", r"hmms?|hidden markov"),
    ("CRF", r"crfs?|conditional random field"),
    # --- tasks ---
    ("Sentiment Analysis", r"sentiment"),
    ("Text Classification", r"text classification|document classification"),
    ("Summarization", r"summari[sz](?:ation|e|er|ing)"),
    ("Machine Translation", r"machine translation|translation"),
    ("Question Answering", r"question[- ]answering"),
    ("Language Modeling", r"language model(?:s|ing|ling)?"),
    # --- neural sequence models ---
    ("RNN", r"rnns?|recurrent neural"),
    ("LSTM", r"lstms?"),
    ("GRU", r"grus?"),
    ("CNN", r"cnns?|convolutional"),
    ("Seq2Seq", r"seq2seq|sequence[- ]to[- ]sequence|encoder[- ]decoder"),
    ("Attention", r"attention"),
    ("Self-Attention", r"self[- ]attention|multi[- ]head"),
    ("Transformer", r"transformers?"),
    ("Positional Encoding", r"positional (?:encoding|embedding)s?"),
    # --- pretrained language models ---
    ("BERT", r"bert"),
    ("RoBERTa", r"roberta"),
    ("DistilBERT", r"distilbert"),
    ("GPT", r"gpt(?:-?\d(?:\.\d)?)?|chatgpt"),
    ("T5", r"t5"),
    ("Masked Language Modeling", r"masked language model(?:s|ing|ling)?|mlm"),
    ("Pre-training", r"pre[- ]?train(?:ed|ing)?"),
    ("Fine-tuning", r"fine[- ]?tun(?:e|ed|ing)"),
    ("Transfer Learning", r"transfer learning"),
    ("LoRA / PEFT", r"lora|peft"),
    ("LLM", r"llms?|large language model"),
    ("Prompting", r"prompt(?:s|ing)?|few[- ]shot|zero[- ]shot|in[- ]context learning"),
    ("Hallucination", r"hallucinat(?:ion|ions|e|es|ed)"),
    ("Hugging Face", r"hugging ?face"),
    # --- evaluation ---
    ("BLEU", r"bleu"),
    ("ROUGE", r"rouge"),
    ("Perplexity", r"perplexity"),
    ("Precision / Recall / F1", r"precision|recall|f1[- ]?score"),
    ("Confusion Matrix", r"confusion matrix"),
    # --- retrieval, agents, graphs ---
    ("RAG", r"rag|retrieval[- ]augmented"),
    ("Retrieval", r"retriev(?:al|er|ers)"),
    ("Vector Store", r"vector (?:database|store|db|index)|faiss|chroma"),
    ("Reranking", r"re-?rank(?:ing|er)?"),
    ("Chunking", r"chunk(?:s|ing)"),
    ("Agents", r"agents?|agentic|smolagents"),
    ("Tool Calling", r"tool (?:call(?:s|ing)?|use)|function calling"),
    ("LangChain", r"langchain|langgraph"),
    ("Knowledge Graph", r"knowledge graphs?"),
    ("Graph Neural Networks", r"gnns?|graph neural|gcn|gat"),
]

_PATTERNS = [
    (label, re.compile(rf"(?<![A-Za-z0-9])(?:{pat})(?![A-Za-z0-9])", re.IGNORECASE))
    for label, pat in CONCEPTS
]

# Graph group per index modality; the app maps groups to colours and labels.
GROUP_OF = {
    "slides": "slides",
    "transcript": "video",
    "notebook": "code",
    "mcq": "quiz",
    "assessment_pdf": "quiz",
    "misc": "other",
    "clarification": "other",
}


@dataclass
class MapNode:
    id: str
    label: str
    group: str  # slides | video | code | quiz | other | concept | query
    week: str | None = None
    value: float = 1.0  # drives node size
    detail: str = ""  # one line shown in the info card
    highlight: bool = False  # ringed: a source cited by the current answer


@dataclass
class MapEdge:
    source: str
    target: str
    weight: float = 1.0
    label: str = ""


@dataclass
class CourseMap:
    nodes: list[MapNode] = field(default_factory=list)
    edges: list[MapEdge] = field(default_factory=list)


def concepts_in(text: str) -> set[str]:
    """Vocabulary concepts mentioned in ``text``."""
    return {label for label, pat in _PATTERNS if pat.search(text)}


def material_label(chunk: Chunk) -> str:
    """Short display name for the file a chunk came from."""
    stem = Path(chunk.source).stem
    if chunk.modality == "transcript":
        m = re.match(r"(W\d+)(?:_L(\d+))?", stem)
        if m:
            lecture = f" Lecture {int(m.group(2))}" if m.group(2) else ""
            return f"{m.group(1)}{lecture} Video"
    return " ".join(stem.replace("_", " ").split())


def _doc_id(source: str) -> str:
    return f"doc:{source}"


def _concept_id(label: str) -> str:
    return f"concept:{label}"


@dataclass
class Corpus:
    """Per-material concept counts, shared by the global and the local map."""

    info: dict[str, MapNode]  # source -> node (value = number of chunks)
    counts: dict[str, Counter]  # source -> concept -> chunks mentioning it
    df: Counter  # concept -> number of materials mentioning it


def scan_corpus(chunks: list[Chunk]) -> Corpus:
    """One pass over the index. Reuse the result: it is the slow part (~1.5 s)."""
    info: dict[str, MapNode] = {}
    counts: dict[str, Counter] = defaultdict(Counter)
    for c in chunks:
        node = info.get(c.source)
        if node is None:
            node = info[c.source] = MapNode(
                id=_doc_id(c.source), label=material_label(c),
                group=GROUP_OF.get(c.modality, "other"), week=c.week, value=0,
            )
        node.value += 1
        counts[c.source].update(concepts_in(c.text))
    df = Counter(concept for per_doc in counts.values() for concept in per_doc)
    return Corpus(info, counts, df)


def _top_concepts(corpus: Corpus, source: str, k: int) -> list[tuple[str, int]]:
    """The ``k`` concepts most characteristic of one material (TF-IDF over materials)."""
    n_docs = max(len(corpus.info), 1)
    n_chunks = corpus.info[source].value
    # A concept has to recur to count, except in very short materials.
    floor = 1 if n_chunks <= 6 else 2
    scored = [
        (count / n_chunks * math.log(1 + n_docs / corpus.df[concept]), concept, count)
        for concept, count in corpus.counts[source].items()
        if count >= floor
    ]
    scored.sort(reverse=True)
    return [(concept, count) for _, concept, count in scored[:k]]


def build_course_map(
    chunks: list[Chunk] | Corpus, concepts_per_material: int = 5, week: str | None = None
) -> CourseMap:
    """Whole-course map: every material linked to its most characteristic concepts.

    ``week`` keeps only that week's materials (concept scores still use the whole course).
    """
    corpus = chunks if isinstance(chunks, Corpus) else scan_corpus(chunks)
    out = CourseMap()
    degree: Counter = Counter()
    for source, base in corpus.info.items():
        if week is not None and base.week != week:
            continue
        top = _top_concepts(corpus, source, concepts_per_material)
        if not top:
            continue  # nothing to connect it with
        node = MapNode(id=base.id, label=base.label, group=base.group, week=base.week,
                       detail=f"{int(base.value)} 个片段")
        out.nodes.append(node)
        for concept, count in top:
            out.edges.append(MapEdge(node.id, _concept_id(concept), float(count)))
            degree[concept] += 1
    for concept, n in degree.items():
        out.nodes.append(MapNode(
            id=_concept_id(concept), label=concept, group="concept",
            value=float(n), detail=f"关联 {n} 份资料",
        ))
    # Size materials by how many concepts they connect, like Obsidian does.
    links = Counter(e.source for e in out.edges)
    for node in out.nodes:
        if node.group != "concept":
            node.value = float(links[node.id])
    return out


def build_query_map(
    query: str,
    hits: list[Chunk],
    all_chunks: list[Chunk] | Corpus,
    max_concepts: int = 10,
    related_per_concept: int = 2,
) -> CourseMap:
    """Map for one question: question -> cited materials -> concepts -> related materials.

    The outer ring is the useful part for a student: other materials that cover
    the same concepts but were not retrieved for this answer.
    """
    corpus = all_chunks if isinstance(all_chunks, Corpus) else scan_corpus(all_chunks)
    out = CourseMap()
    qid = "query"
    out.nodes.append(MapNode(id=qid, label="当前问题", group="query", value=6, detail=query))

    cited: dict[str, MapNode] = {}
    concept_hits: Counter = Counter()
    concept_edges: dict[tuple[str, str], int] = Counter()
    for c in hits:
        if c.source not in corpus.info:
            continue
        if c.source not in cited:
            base = corpus.info[c.source]
            cited[c.source] = MapNode(
                id=base.id, label=base.label, group=base.group, week=base.week,
                value=3, detail="被本次回答引用", highlight=True,
            )
            out.edges.append(MapEdge(qid, base.id, 2.0))
        for concept in concepts_in(c.text):
            concept_hits[concept] += 1
            concept_edges[(c.source, concept)] += 1
    out.nodes.extend(cited.values())

    keep = [concept for concept, _ in concept_hits.most_common(max_concepts)]
    related: dict[str, MapNode] = {}
    for concept in keep:
        cid = _concept_id(concept)
        out.nodes.append(MapNode(
            id=cid, label=concept, group="concept", value=1 + concept_hits[concept],
            detail=f"课程中 {corpus.df[concept]} 份资料涉及",
        ))
        for (source, c2), n in concept_edges.items():
            if c2 == concept:
                out.edges.append(MapEdge(_doc_id(source), cid, float(n)))
        # Other materials that cover this concept most heavily.
        others = sorted(
            ((corpus.counts[s][concept], s) for s in corpus.info
             if s not in cited and corpus.counts[s][concept] >= 2),
            reverse=True,
        )[:related_per_concept]
        for n, source in others:
            base = corpus.info[source]
            if source not in related:
                related[source] = MapNode(
                    id=base.id, label=base.label, group=base.group, week=base.week,
                    value=1, detail="相关资料（未被引用）",
                )
            out.edges.append(MapEdge(base.id, cid, float(n)))
    out.nodes.extend(related.values())
    return out


def triples_to_map(triples) -> CourseMap:
    """Wrap LLM / co-occurrence triples so they render in the same style."""
    out = CourseMap()
    degree: Counter = Counter()
    seen: dict[tuple[str, str], MapEdge] = {}
    for t in triples:
        if not t.head or not t.tail or t.head == t.tail:
            continue
        key = tuple(sorted((t.head, t.tail)))
        if key in seen:
            seen[key].weight += t.weight
            continue
        seen[key] = MapEdge(f"concept:{t.head}", f"concept:{t.tail}", t.weight,
                            t.relation.replace("_", " "))
        degree[t.head] += 1
        degree[t.tail] += 1
    out.edges = list(seen.values())
    out.nodes = [
        MapNode(id=f"concept:{name}", label=name, group="concept", value=float(n),
                detail=f"{n} 条关系")
        for name, n in degree.items()
    ]
    return out
