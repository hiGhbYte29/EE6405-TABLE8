"""Quiz benchmark, step 1: turn the test sets into model-agnostic jobs.

    python -m ee6405.eval.bench                 # writes reports/bench/jobs.jsonl

A *job* is one question under one retrieval configuration, already rendered as
chat messages. Retrieval happens here, once, so every model answers from
exactly the same context -- and so the answering step (``scripts/run_bench.py``)
needs nothing but the standard library and can run on any machine that can
reach a model endpoint.

Configurations
--------------
``closed``                 no retrieval: the plain-LLM baseline
``bm25@5 dense@5 hybrid@5``  the three retrievers at the app's default k
``hybrid@1 @3 @10``        top-k sensitivity
``hybrid@5:slides`` ...    retrieval restricted to one kind of course material

Leak guard: practice MCQs and graded quizzes are themselves in the index (with
their answers). They are excluded from retrieval in every configuration, and
every job records how many assessment chunks it retrieved -- which must be 0.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ee6405.config import REPO_ROOT, settings
from ee6405.eval.quiz_sets import OUT_DIR, QuizQuestion, load_sets
from ee6405.rag.embedding import build_embedder
from ee6405.rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from ee6405.rag.store import ChunkStore

BENCH_DIR = REPO_ROOT / "reports" / "bench"
LEAKY = ["mcq", "assessment_pdf"]  # never retrievable during evaluation
MAX_CHUNK_CHARS = 1500

MAIN = ["closed", "bm25@5", "dense@5", "hybrid@5"]
TOPK = ["hybrid@1", "hybrid@3", "hybrid@10"]
SOURCES = ["hybrid@5:slides", "hybrid@5:transcript", "hybrid@5:notebook"]

SYSTEM = (
    "You are taking a quiz for NTU course EE6405 (Natural Language Processing). "
    "Answer from the course material when it is provided and relevant; "
    "otherwise use your own knowledge."
)
SINGLE = (
    "Exactly one option is correct. Finish your reply with a line of the form\n"
    "ANSWER: <letter>"
)
MULTI = (
    "One or more options are correct. Select every correct option and no incorrect one. "
    "Finish your reply with a line of the form\nANSWER: <letters separated by commas>"
)


def parse_config(config: str) -> tuple[str | None, int, str | None]:
    """``"hybrid@5:slides"`` -> ``("hybrid", 5, "slides")``; ``"closed"`` -> ``(None, 0, None)``."""
    if config == "closed":
        return None, 0, None
    head, _, modality = config.partition(":")
    kind, _, k = head.partition("@")
    return kind, int(k or settings.top_k), modality or None


def render_question(q: QuizQuestion) -> str:
    options = "\n".join(f"{letter}. {text}" for letter, text in q.options.items())
    return f"{q.stem}\n\n{options}"


def build_messages(q: QuizQuestion, context: list[str]) -> list[dict[str, str]]:
    parts = []
    if context:
        blocks = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(context, 1))
        parts.append(f"COURSE MATERIAL:\n{blocks}")
    parts.append(f"QUESTION:\n{render_question(q)}")
    parts.append(MULTI if q.multi else SINGLE)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_jobs(sets: dict[str, list[QuizQuestion]], configs: list[str], index_path: Path) -> list[dict]:
    store = ChunkStore(index_path, read_only=True)
    needs_dense = any(parse_config(c)[0] in {"dense", "hybrid"} for c in configs)
    embedder = build_embedder() if needs_dense else None  # one model load, shared
    retrievers = {"bm25": BM25Retriever(store)}
    if needs_dense:
        retrievers["dense"] = DenseRetriever(store, embedder)
        retrievers["hybrid"] = HybridRetriever(store, embedder)

    jobs = []
    for dataset, questions in sets.items():
        for q in questions:
            query = render_question(q)
            for config in configs:
                kind, k, modality = parse_config(config)
                hits = []
                if kind:
                    hits = retrievers[kind].retrieve(
                        query, k, modality=modality, exclude_modality=LEAKY
                    )
                jobs.append({
                    "job_id": f"{q.qid}|{config}",
                    "qid": q.qid,
                    "dataset": dataset,
                    "config": config,
                    "messages": build_messages(q, [h.chunk.text[:MAX_CHUNK_CHARS] for h in hits]),
                    "retrieved": [
                        {"source": h.chunk.source, "modality": h.chunk.modality,
                         "week": h.chunk.week, "locator": h.chunk.locator}
                        for h in hits
                    ],
                    "leaked": sum(h.chunk.modality in LEAKY for h in hits),
                })
    store.close()
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(description="Render quiz questions into benchmark jobs.")
    p.add_argument("--sets", type=Path, default=OUT_DIR)
    p.add_argument("--out", type=Path, default=BENCH_DIR / "jobs.jsonl")
    p.add_argument("--configs", nargs="+", default=MAIN + TOPK + SOURCES)
    args = p.parse_args()

    sets = load_sets(args.sets)
    if not sets:
        raise SystemExit(f"no test sets in {args.sets}; run `python -m ee6405.eval.quiz_sets` first")
    jobs = build_jobs(sets, args.configs, settings.index_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs), encoding="utf-8")
    questions = {name: [asdict(q) for q in qs] for name, qs in sets.items()}
    (args.out.parent / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    leaked = sum(j["leaked"] for j in jobs)
    n_q = sum(len(qs) for qs in sets.values())
    print(f"{len(jobs)} jobs = {n_q} questions x {len(args.configs)} configs -> {args.out}")
    print(f"assessment chunks retrieved: {leaked} (must be 0)")
    if leaked:
        raise SystemExit("leak guard failed")


if __name__ == "__main__":
    main()
