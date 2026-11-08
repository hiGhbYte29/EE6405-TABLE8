"""Run the configured systems over the eval set and write a results table.

    python -m ee6405.eval.run_eval --config configs/eval.yaml

Each row of the output is one (system, question) pair; the printed summary is
the per-system mean. Two comparisons the report needs come out of the same run:

* **baselines**  -- our RAG system vs plain LLMs (different models / endpoints)
* **ablations**  -- base vs without-rag vs without-memory, and bm25 vs dense
                    vs hybrid retrieval
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ee6405.agent.chain import StudyAssistant
from ee6405.agent.llm import build_llm
from ee6405.agent.memory import ConversationMemory
from ee6405.config import REPO_ROOT, settings
from ee6405.eval.metrics import (
    answer_token_f1,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ee6405.rag.retriever import build_retriever
from ee6405.rag.store import ChunkStore


@dataclass
class EvalExample:
    question: str
    relevant_sources: list[str]
    reference_answer: str = ""
    qid: str = ""


def load_eval_set(path: str | Path) -> list[EvalExample]:
    """Read a JSONL file of eval examples.

    One JSON object per line::

        {"qid": "q1",
         "question": "What is the difference between BERT and RoBERTa?",
         "relevant_sources": ["data/raw/slides/week6.pdf"],
         "reference_answer": "RoBERTa drops NSP and trains longer ..."}
    """
    path = Path(path)
    examples: list[EvalExample] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            row = json.loads(line)
            examples.append(
                EvalExample(
                    question=row["question"],
                    relevant_sources=row.get("relevant_sources", []),
                    reference_answer=row.get("reference_answer", ""),
                    qid=row.get("qid", f"q{i}"),
                )
            )
    return examples


def evaluate_system(
    name: str,
    spec: dict[str, Any],
    examples: list[EvalExample],
    index_path: Path,
    k: int,
) -> list[dict[str, Any]]:
    """Run one system spec over every example and return per-question rows."""
    store = ChunkStore(index_path)
    retriever = (
        build_retriever(store, spec.get("retriever", settings.retriever))
        if spec.get("use_rag", True)
        else None
    )
    memory = (
        ConversationMemory(settings.memory_path, session_id=f"eval::{name}")
        if spec.get("use_memory", True)
        else None
    )
    if memory is not None:
        memory.clear()  # every system starts from the same blank slate

    llm = build_llm(model=spec.get("model"))
    assistant = StudyAssistant(retriever=retriever, llm=llm, memory=memory)

    rows: list[dict[str, Any]] = []
    for ex in examples:
        answer = assistant.answer(
            ex.question,
            top_k=k,
            use_rag=spec.get("use_rag", True),
            use_memory=spec.get("use_memory", True),
        )
        retrieved = [h.chunk.source for h in answer.hits]
        rows.append(
            {
                "system": name,
                "qid": ex.qid,
                "question": ex.question,
                "model": answer.model,
                "retriever": spec.get("retriever", "-") if spec.get("use_rag", True) else "-",
                "use_rag": spec.get("use_rag", True),
                "use_memory": spec.get("use_memory", True),
                f"recall@{k}": recall_at_k(retrieved, ex.relevant_sources, k),
                f"precision@{k}": precision_at_k(retrieved, ex.relevant_sources, k),
                f"hit@{k}": hit_rate_at_k(retrieved, ex.relevant_sources, k),
                "mrr": mean_reciprocal_rank(retrieved, ex.relevant_sources),
                f"ndcg@{k}": ndcg_at_k(retrieved, ex.relevant_sources, k),
                "answer_f1": (
                    answer_token_f1(answer.text, ex.reference_answer)
                    if ex.reference_answer
                    else float("nan")
                ),
                "answer": answer.text,
            }
        )

    store.close()
    if memory is not None:
        memory.close()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate baselines and ablations.")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "eval.yaml")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "eval_results.csv")
    p.add_argument("--only", nargs="*", help="run only these system names")
    p.add_argument("--mcq", action="store_true", help="evaluate indexed MCQs with mcq chunks excluded from retrieval")
    p.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    k = int(cfg.get("top_k", settings.top_k))
    index_path = Path(cfg.get("index_path", settings.index_path))
    if args.mcq:
        print_mcq_results(evaluate_mcq(index_path, args.retriever, k))
        return
    examples = load_eval_set(REPO_ROOT / cfg["eval_set"])
    if not examples:
        raise SystemExit(f"eval set {cfg['eval_set']} is empty -- write some questions first")

    systems = cfg["systems"]
    if args.only:
        systems = {n: s for n, s in systems.items() if n in args.only}

    rows: list[dict[str, Any]] = []
    for name, spec in systems.items():
        print(f"--> {name}")
        rows.extend(evaluate_system(name, spec or {}, examples, index_path, k))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    metric_cols = [c for c in df.columns if c.startswith(("recall@", "precision@", "hit@", "ndcg@")) or c in {"mrr", "answer_f1"}]
    summary = df.groupby("system")[metric_cols].mean(numeric_only=True).round(4)
    print()
    print(summary.to_string())
    print(f"\nper-question rows -> {args.out}")


if __name__ == "__main__":
    main()

@dataclass
class MCQResult:
    qid: str
    question: str
    predicted: str | None
    correct: str | None
    is_correct: bool
    retrieved: list[dict[str, str]]


def _predict_option(text: str) -> str | None:
    """Extract a 1-5 option from an LLM response, if it states one."""
    import re
    m = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*([1-5])\b", text, re.I)
    return m.group(1) if m else (re.search(r"\b([1-5])\b", text).group(1) if re.search(r"\b([1-5])\b", text) else None)


def evaluate_mcq(index_path: str | Path, retriever_kind: str = "hybrid", top_k: int = 5, llm=None) -> list[MCQResult]:
    """Answer indexed MCQs while *always* excluding mcq chunks from retrieval."""
    store = ChunkStore(index_path)
    questions = [c for c in store.all_chunks() if c.modality == "mcq"]
    retriever = build_retriever(store, retriever_kind)
    llm = llm or build_llm()
    results: list[MCQResult] = []
    for q in questions:
        hits = retriever.retrieve(q.text, top_k, exclude_modality="mcq")
        context = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1))
        prompt = [{"role": "user", "content": "Answer this multiple-choice question with the option number only.\nQUESTION:\n" + q.text + "\nCOURSE CONTEXT:\n" + context}]
        predicted = _predict_option(llm.chat(prompt))
        correct = str(q.metadata.get("answer")) if q.metadata.get("answer") is not None else None
        results.append(MCQResult(q.metadata.get("qid", q.chunk_id), q.text, predicted, correct, predicted == correct and correct is not None, [{"source": h.chunk.source, "locator": h.chunk.locator} for h in hits]))
    store.close()
    return results


def print_mcq_results(results: list[MCQResult]) -> None:
    for r in results:
        print(f"{r.qid}: predicted={r.predicted or '-'} correct={r.correct or '-'} {'OK' if r.is_correct else 'WRONG'}")
    accuracy = sum(r.is_correct for r in results) / len(results) if results else 0.0
    print(f"accuracy: {accuracy:.4f} ({sum(r.is_correct for r in results)}/{len(results)})")
