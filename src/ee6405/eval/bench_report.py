"""Quiz benchmark, step 3: score the answers, write tables, draw figures.

    python -m ee6405.eval.bench_report          # reads reports/bench/, writes tables + figures

Scoring
-------
A reply is scored on its final ``ANSWER:`` line. No parsable line counts as
wrong (and is reported as ``unparsed``), so a model cannot gain by rambling.

* single-answer sets (practice, coding): accuracy
* multi-answer set (theory): exact match, plus option-level precision /
  recall / F1 -- picking 2 of 3 correct options is not the same as picking none
* every accuracy carries a Wilson 95% interval; RAG vs closed-book is tested
  per model with an exact McNemar test on the paired questions
* retrieval is scored without relevance labels by a proxy: does a retrieved
  chunk come from the week the question was set for?
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from ee6405.config import REPO_ROOT
from ee6405.eval.bench import BENCH_DIR, MAIN, SOURCES, parse_config

FIG_DIR = REPO_ROOT / "reports" / "figures"
DATASETS = ["practice", "coding", "theory", "course"]
DATASET_LABEL = {
    "practice": "Practice MCQ",
    "coding": "Coding quiz",
    "theory": "Theory quiz (multi-answer)",
    "course": "Course-specific facts",
}

# ------------------------------------------------------------------- scoring

_ANSWER = re.compile(r"ANSWER\s*[:：]\s*(.*)", re.I)


def parse_answer(text: str) -> list[str] | None:
    """Letters on the last ``ANSWER:`` line, or None when there is none."""
    found = _ANSWER.findall(text or "")
    if not found:
        # A reply that is nothing but option letters ("A, C") is unambiguous too.
        bare = re.fullmatch(r"\s*[A-E](?:[\s,，、/&]+[A-E])*\s*[.。]?\s*", (text or "").replace("*", ""))
        if not bare:
            return None
        found = [bare.group(0)]
    letters = re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", found[-1].replace("*", ""))
    return sorted(set(letters)) or None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_exact(helped: int, hurt: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs."""
    n = helped + hurt
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(helped, hurt) + 1)) / 2**n
    return min(1.0, 2 * tail)


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def load(bench_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-answer rows and per-job retrieval rows."""
    questions = {
        q["qid"]: q
        for qs in json.loads((bench_dir / "questions.json").read_text(encoding="utf-8")).values()
        for q in qs
    }
    jobs = {}
    for line in (bench_dir / "jobs.jsonl").read_text(encoding="utf-8").splitlines():
        j = json.loads(line)
        jobs[j["job_id"]] = j

    answers: dict[tuple[str, str], dict] = {}
    for path in sorted(bench_dir.glob("responses_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "error" in r or r["job_id"] not in jobs:
                continue
            answers[(r["job_id"], r["model"])] = r  # a later retry replaces an earlier row

    rows = []
    for (job_id, model), r in answers.items():
        job, q = jobs[job_id], questions[jobs[job_id]["qid"]]
        gold, pred = set(q["correct"]), parse_answer(r.get("text", ""))
        chosen = set(pred or [])
        rows.append({
            "model": model, "config": job["config"], "dataset": job["dataset"], "qid": job["qid"],
            "week": q.get("week") or "", "gold": "".join(sorted(gold)), "pred": "".join(sorted(chosen)),
            "unparsed": pred is None, "correct": pred is not None and chosen == gold,
            "tp": len(chosen & gold), "fp": len(chosen - gold), "fn": len(gold - chosen),
            "latency": r.get("latency"), "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
        })

    retrieval = []
    for job in jobs.values():
        kind, k, modality = parse_config(job["config"])
        if not kind:
            continue
        q = questions[job["qid"]]
        row = {"config": job["config"], "dataset": job["dataset"], "qid": job["qid"]}
        if q.get("week"):  # proxy relevance: the chunk is from the question's week
            hits = [h.get("week") == q["week"] for h in job["retrieved"]]
            first = next((i for i, h in enumerate(hits, 1) if h), None)
            row.update(week_hit=any(hits), week_precision=sum(hits) / max(len(hits), 1),
                       week_mrr=1 / first if first else 0.0)
        if job["dataset"] == "course":  # true relevance: the chunk the question was written from
            page = job["qid"].rsplit("_", 1)[-1]  # "page12" / "block3"
            docs = [h["source"] == q["source"] for h in job["retrieved"]]
            chunks = [d and h.get("locator", "").replace(":", "") == page
                      for d, h in zip(docs, job["retrieved"], strict=True)]
            first = next((i for i, h in enumerate(chunks, 1) if h), None)
            row.update(gold_doc_hit=any(docs), gold_chunk_hit=any(chunks),
                       gold_chunk_mrr=1 / first if first else 0.0)
        retrieval.append(row)
    return pd.DataFrame(rows), pd.DataFrame(retrieval)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, config, dataset), g in df.groupby(["model", "config", "dataset"]):
        n, k = len(g), int(g["correct"].sum())
        lo, hi = wilson(k, n)
        tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out.append({
            "model": model, "config": config, "dataset": dataset, "n": n, "correct": k,
            "accuracy": k / n, "ci_low": lo, "ci_high": hi,
            "option_precision": precision, "option_recall": recall,
            "option_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "unparsed": int(g["unparsed"].sum()),
            "median_latency_s": g["latency"].median(), "mean_prompt_tokens": g["prompt_tokens"].mean(),
        })
    return pd.DataFrame(out)


def overall(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per model and config: pooled over questions, and macro over datasets."""
    out = []
    for (model, config), g in summary.groupby(["model", "config"]):
        if set(g["dataset"]) != set(DATASETS):
            continue
        n, k = int(g["n"].sum()), int(g["correct"].sum())
        lo, hi = wilson(k, n)
        out.append({"model": model, "config": config, "n": n, "correct": k, "accuracy": k / n,
                    "ci_low": lo, "ci_high": hi, "macro_accuracy": g["accuracy"].mean()})
    return pd.DataFrame(out)


def paired_tests(df: pd.DataFrame, baseline: str = "closed") -> pd.DataFrame:
    out = []
    for model, g in df.groupby("model"):
        wide = g.pivot_table(index="qid", columns="config", values="correct", aggfunc="first")
        if baseline not in wide:
            continue
        for config in wide.columns:
            if config == baseline:
                continue
            pair = wide[[baseline, config]].dropna().astype(bool)
            helped = int((~pair[baseline] & pair[config]).sum())
            hurt = int((pair[baseline] & ~pair[config]).sum())
            out.append({
                "model": model, "config": config, "n": len(pair),
                "baseline_acc": pair[baseline].mean(), "config_acc": pair[config].mean(),
                "delta": pair[config].mean() - pair[baseline].mean(),
                "helped": helped, "hurt": hurt, "mcnemar_p": mcnemar_exact(helped, hurt),
            })
    return pd.DataFrame(out)


# ------------------------------------------------------------------- figures

INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SECONDARY = "#52514e"
BASELINE = "#898781"  # closed-book: the neutral reference, not a series colour
RAG = "#2a78d6"
RETRIEVER = {"bm25@5": "#2a78d6", "dense@5": "#eb6834", "hybrid@5": "#1baf7a"}
RETRIEVER_LABEL = {"closed": "No retrieval", "bm25@5": "BM25", "dense@5": "Dense (BGE-m3)",
                   "hybrid@5": "Hybrid (RRF)"}
HELPED, HURT = "#2a78d6", "#eb6834"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Segoe UI", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 11, "text.color": INK, "axes.labelcolor": SECONDARY,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })
    return plt


def _title(fig, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.01, y=1.0, ha="left", va="bottom", fontsize=14.5, fontweight="semibold")
    fig.text(0.01, 0.985, subtitle, ha="left", va="top", fontsize=10.5, color=SECONDARY)


def _pct(ax, axis: str = "y") -> None:
    from matplotlib.ticker import PercentFormatter

    getattr(ax, f"{axis}axis").set_major_formatter(PercentFormatter(1.0, decimals=0))


def fig_rag_effect(summary: pd.DataFrame, path: Path, rag: str = "hybrid@5") -> None:
    """Dumbbell per model and dataset: closed-book -> with RAG."""
    plt = _style()
    wide = summary.pivot_table(index=["model", "dataset"], columns="config", values="accuracy")
    if "closed" not in wide or rag not in wide:
        return
    order = (wide["closed"].groupby("model").mean()).sort_values().index.tolist()
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.3 * len(DATASETS), 0.52 * len(order) + 2.4), sharey=True)
    for ax, dataset in zip(axes, DATASETS, strict=False):
        n = int(summary[(summary.dataset == dataset)]["n"].max())
        for y, model in enumerate(order):
            if (model, dataset) not in wide.index:
                continue
            a, b = wide.loc[(model, dataset), "closed"], wide.loc[(model, dataset), rag]
            ax.plot([a, b], [y, y], color=RAG if b >= a else HURT, lw=2, alpha=0.45, zorder=2,
                    solid_capstyle="round")
            ax.scatter([a], [y], s=62, color=BASELINE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.scatter([b], [y], s=62, color=RAG, zorder=4, edgecolor=SURFACE, linewidth=1.5)
            ax.annotate(f"{(b - a) * 100:+.0f}", (max(a, b), y), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=9.5, color=SECONDARY)
        ax.set_title(f"{DATASET_LABEL[dataset]}  ·  {n} questions")
        ax.set_xlim(0.0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        _pct(ax, "x")
        ax.grid(axis="y", visible=False)
        ax.set_yticks(range(len(order)), order)
    axes[0].scatter([], [], s=62, color=BASELINE, label="No retrieval (model alone)")
    axes[0].scatter([], [], s=62, color=RAG, label="With RAG (hybrid, top 5)")
    fig.legend(loc="lower left", bbox_to_anchor=(0.01, -0.07), ncols=2, fontsize=10.5)
    _title(fig, "What retrieval adds, model by model",
           "Share of questions answered exactly right. Number = change in percentage points.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def fig_retrievers(summary: pd.DataFrame, path: Path) -> None:
    """Mean accuracy over models, per dataset, for each retriever."""
    plt = _style()
    main = summary[summary.config.isin(MAIN)]
    full = main.groupby("model")["config"].nunique()
    main = main[main.model.isin(full[full == len(MAIN)].index)]
    if main.empty:
        return
    mean = main.groupby(["dataset", "config"])["accuracy"].mean().unstack()[MAIN].loc[DATASETS]
    n_models = main.model.nunique()
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    width = 0.19
    for i, config in enumerate(MAIN):
        xs = [x + (i - 1.5) * (width + 0.015) for x in range(len(DATASETS))]
        color = BASELINE if config == "closed" else RETRIEVER[config]
        bars = ax.bar(xs, mean[config], width, color=color, label=RETRIEVER_LABEL[config], zorder=3)
        for bar, v in zip(bars, mean[config], strict=False):
            ax.annotate(f"{v * 100:.0f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9.5, color=SECONDARY)
    ax.set_xticks(range(len(DATASETS)), [DATASET_LABEL[d] for d in DATASETS])
    ax.set_ylim(0, 1.0)
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=4, fontsize=10.5)
    _title(fig, "Which retriever helps most",
           f"Exact-match accuracy, averaged over {n_models} models. Top 5 chunks per question.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def fig_topk(summary: pd.DataFrame, path: Path, model: str) -> None:
    plt = _style()
    g = summary[(summary.model == model) & (summary.config.str.fullmatch(r"closed|hybrid@\d+"))].copy()
    if g.config.nunique() < 3:
        return
    g["k"] = g.config.map(lambda c: parse_config(c)[1])
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.1 * len(DATASETS), 3.9), sharey=True)
    for ax, dataset in zip(axes, DATASETS, strict=False):
        d = g[g.dataset == dataset].sort_values("k")
        ax.fill_between(d.k, d.ci_low, d.ci_high, color=RAG, alpha=0.10, lw=0)
        ax.plot(d.k, d.accuracy, color=RAG, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=3)
        for k, v in zip(d.k, d.accuracy, strict=False):
            ax.annotate(f"{v * 100:.0f}", (k, v), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9.5, color=SECONDARY)
        ax.set_title(DATASET_LABEL[dataset])
        ax.set_xticks(sorted(d.k), ["none" if k == 0 else str(k) for k in sorted(d.k)])
        ax.set_xlabel("Chunks retrieved (top-k)")
        ax.set_ylim(0, 1.08)
        _pct(ax)
        ax.grid(axis="x", visible=False)
    _title(fig, "How much context is enough",
           f"{model}, hybrid retrieval. Band = Wilson 95% interval.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def fig_sources(summary: pd.DataFrame, path: Path, model: str) -> None:
    plt = _style()
    configs = ["closed", *SOURCES, "hybrid@5"]
    labels = ["No retrieval", "Slides only", "Video transcripts only", "Notebooks only", "All sources"]
    g = summary[(summary.model == model) & summary.config.isin(configs)]
    if g.config.nunique() < len(configs):
        return
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.1 * len(DATASETS), 3.7), sharex=True)
    for ax, dataset in zip(axes, DATASETS, strict=False):
        d = g[g.dataset == dataset].set_index("config").loc[configs]
        colors = [BASELINE, *["#86b6ef"] * 3, RAG]
        bars = ax.barh(range(len(configs)), d.accuracy, 0.62, color=colors, zorder=3)
        for bar, v in zip(bars, d.accuracy, strict=False):
            ax.annotate(f"{v * 100:.0f}%", (v, bar.get_y() + bar.get_height() / 2), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=9.5, color=SECONDARY)
        ax.set_yticks(range(len(configs)), labels if ax is axes[0] else [""] * len(configs))
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1.0])
        _pct(ax, "x")
        ax.grid(axis="y", visible=False)
        ax.set_title(DATASET_LABEL[dataset])
    _title(fig, "Which course material carries the answers",
           f"{model}, hybrid retrieval restricted to one kind of material at a time.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def fig_flips(tests: pd.DataFrame, path: Path, rag: str = "hybrid@5") -> None:
    plt = _style()
    t = tests[tests.config == rag].sort_values("delta")
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(t) + 2.0))
    ys = range(len(t))
    ax.barh(ys, -t.hurt, 0.6, color=HURT, label="Right alone, wrong with RAG", zorder=3)
    ax.barh(ys, t.helped, 0.6, color=HELPED, label="Wrong alone, right with RAG", zorder=3)
    for y, (_, r) in zip(ys, t.iterrows(), strict=False):
        ax.annotate(str(r.helped), (r.helped, y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=SECONDARY)
        ax.annotate(str(r.hurt), (-r.hurt, y), xytext=(-5, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=9.5, color=SECONDARY)
    ax.set_yticks(ys, [f"{m}   ({fmt_p(p)})" for m, p in zip(t.model, t.mcnemar_p, strict=False)])
    ax.axvline(0, color=AXIS, lw=1)
    lim = max(t.helped.max(), t.hurt.max()) + 4
    ax.set_xlim(-lim, lim)
    ax.set_xticks([x for x in ax.get_xticks() if abs(x) <= lim],
                  [str(int(abs(x))) for x in ax.get_xticks() if abs(x) <= lim])
    ax.set_xlabel("Questions whose outcome changed")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=10.5)
    _title(fig, "Questions RAG fixed, and questions it broke",
           f"All {int(t.n.max())} questions, hybrid top 5 vs no retrieval. p = exact McNemar test.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def fig_multilabel(summary: pd.DataFrame, path: Path, rag: str = "hybrid@5") -> None:
    plt = _style()
    g = summary[(summary.dataset == "theory") & summary.config.isin(["closed", rag])]
    both = g.groupby("model")["config"].nunique()
    g = g[g.model.isin(both[both == 2].index)]
    if g.empty:
        return
    metrics = ["accuracy", "option_precision", "option_recall", "option_f1"]
    names = ["Exact match", "Option precision", "Option recall", "Option F1"]
    mean = g.groupby("config")[metrics].mean()
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    width = 0.34
    for i, (config, color, label) in enumerate([("closed", BASELINE, "No retrieval"),
                                                (rag, RAG, "With RAG (hybrid, top 5)")]):
        xs = [x + (i - 0.5) * (width + 0.02) for x in range(len(metrics))]
        bars = ax.bar(xs, mean.loc[config], width, color=color, label=label, zorder=3)
        for bar, v in zip(bars, mean.loc[config], strict=False):
            ax.annotate(f"{v * 100:.0f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9.5, color=SECONDARY)
    ax.set_xticks(range(len(metrics)), names)
    ax.set_ylim(0, 1.08)
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=2, fontsize=10.5)
    _title(fig, "Multi-answer theory questions: one metric is not enough",
           f"Mean over {g.model.nunique()} models. Exact match needs every option right; "
           "F1 gives credit per option.")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


def fig_retrieval(retrieval: pd.DataFrame, path: Path) -> None:
    """Retrieval quality on the one set with true labels: the chunk each question came from."""
    plt = _style()
    if "gold_chunk_hit" not in retrieval:
        return
    r = retrieval[retrieval.dataset == "course"]
    at5 = r[r.config.isin(list(RETRIEVER))].groupby("config")[
        ["gold_chunk_hit", "gold_doc_hit", "gold_chunk_mrr"]].mean().loc[list(RETRIEVER)]
    by_k = r[r.config.str.fullmatch(r"hybrid@\d+")].copy()
    by_k["k"] = by_k.config.map(lambda c: parse_config(c)[1])
    by_k = by_k.groupby("k")["gold_chunk_hit"].mean()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3), width_ratios=[1.7, 1])
    names = ["Exact page in top 5", "Right document in top 5", "MRR of the exact page"]
    width = 0.24
    for i, config in enumerate(RETRIEVER):
        xs = [x + (i - 1) * (width + 0.015) for x in range(3)]
        bars = ax.bar(xs, at5.loc[config], width, color=RETRIEVER[config],
                      label=RETRIEVER_LABEL[config], zorder=3)
        for bar, v in zip(bars, at5.loc[config], strict=True):
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9.5, color=SECONDARY)
    ax.set_xticks(range(3), names)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="x", visible=False)
    ax.set_title("Three retrievers, top 5")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3, fontsize=10.5)

    ax2.plot(by_k.index, by_k.values, color=RETRIEVER["hybrid@5"], lw=2, marker="o", ms=7,
             mec=SURFACE, mew=1.5, zorder=3)
    for k, v in by_k.items():
        ax2.annotate(f"{v:.2f}", (k, v), xytext=(0, 8), textcoords="offset points", ha="center",
                     fontsize=9.5, color=SECONDARY)
    ax2.set_xticks(list(by_k.index))
    ax2.set_xlabel("Chunks retrieved (top-k)")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Hybrid: exact page found, by k")
    _title(fig, "Does retrieval find the page the answer is on?",
           f"{r.qid.nunique()} course-specific questions, each written from one known page of course material.")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


def write_markdown(df: pd.DataFrame, summary: pd.DataFrame, totals: pd.DataFrame,
                   tests: pd.DataFrame, retrieval: pd.DataFrame, path: Path) -> None:
    """Results tables as markdown, generated so no number is ever copied by hand."""

    def pct(v: float) -> str:
        return f"{v * 100:.1f}"

    def table(frame: pd.DataFrame) -> str:
        head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
        rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
        return "\n".join([head, rule, *body])

    main = summary[summary.config.isin(MAIN)]
    complete = main.groupby("model")["config"].nunique()
    models = sorted(complete[complete == len(MAIN)].index)
    sizes = summary.groupby("dataset")["n"].max()
    out = ["# Quiz benchmark results", "",
           "Generated by `python -m ee6405.eval.bench_report`. Do not edit by hand.", "",
           "Accuracy is exact match: for multi-answer theory questions every option must be right.", ""]

    out += ["## Test sets", "",
            table(pd.DataFrame({"set": [DATASET_LABEL[d] for d in DATASETS if d in sizes],
                                "questions": [int(sizes[d]) for d in DATASETS if d in sizes]})), ""]

    out += ["## Accuracy by model: no retrieval vs RAG (hybrid, top 5)", ""]
    rows = []
    for model in models:
        row = {"model": model}
        for dataset in [d for d in DATASETS if d in sizes]:
            g = main[(main.model == model) & (main.dataset == dataset)].set_index("config")
            row[DATASET_LABEL[dataset]] = f"{pct(g.accuracy['closed'])} → {pct(g.accuracy['hybrid@5'])}"
        t = totals[(totals.model == model)].set_index("config")
        row["all questions"] = f"{pct(t.accuracy['closed'])} → {pct(t.accuracy['hybrid@5'])}"
        x = tests[(tests.model == model) & (tests.config == "hybrid@5")].iloc[0]
        row["fixed / broken"] = f"{x.helped} / {x.hurt}"
        row["McNemar"] = fmt_p(x.mcnemar_p)
        rows.append(row)
    out += [table(pd.DataFrame(rows)), ""]

    out += [f"## Retrievers, mean over {len(models)} models", ""]
    mean = main[main.model.isin(models)].groupby(["dataset", "config"])["accuracy"].mean().unstack()
    rows = [{"set": DATASET_LABEL[d], **{RETRIEVER_LABEL[c]: pct(mean.loc[d, c]) for c in MAIN}}
            for d in DATASETS if d in mean.index]
    pooled = totals[totals.model.isin(models) & totals.config.isin(MAIN)].groupby("config")["accuracy"].mean()
    rows.append({"set": "All questions (pooled)", **{RETRIEVER_LABEL[c]: pct(pooled[c]) for c in MAIN}})
    out += [table(pd.DataFrame(rows)), ""]

    theory = main[(main.dataset == "theory") & main.model.isin(models)]
    if not theory.empty:
        out += ["## Multi-answer theory questions: option-level scores, mean over models", ""]
        m = theory.groupby("config")[["accuracy", "option_precision", "option_recall", "option_f1"]].mean()
        out += [table(pd.DataFrame([{"config": RETRIEVER_LABEL[c], "exact match": pct(m.accuracy[c]),
                                     "option precision": pct(m.option_precision[c]),
                                     "option recall": pct(m.option_recall[c]),
                                     "option F1": pct(m.option_f1[c])} for c in MAIN])), ""]

    sweeps = summary[~summary.config.isin(MAIN)]
    for model in sorted(sweeps.model.unique()):
        g = summary[summary.model == model]
        wide = g.pivot_table(index="config", columns="dataset", values="accuracy")
        order = [c for c in ["closed", "hybrid@1", "hybrid@3", "hybrid@5", "hybrid@10", *SOURCES] if c in wide.index]
        out += [f"## Top-k and single-source ablations ({model})", ""]
        out += [table(pd.DataFrame([{"config": c, **{DATASET_LABEL[d]: pct(wide.loc[c, d])
                                                     for d in DATASETS if d in wide.columns}}
                                    for c in order])), ""]

    if "gold_chunk_hit" in retrieval:
        r = retrieval[retrieval.dataset == "course"].groupby("config")[
            ["gold_chunk_hit", "gold_doc_hit", "gold_chunk_mrr"]].mean()
        order = [c for c in ["bm25@5", "dense@5", "hybrid@1", "hybrid@3", "hybrid@5", "hybrid@10"] if c in r.index]
        out += ["## Retrieval quality on the course-specific set (true labels)", "",
                table(pd.DataFrame([{"config": c, "exact page retrieved": f"{r.gold_chunk_hit[c]:.2f}",
                                     "right document retrieved": f"{r.gold_doc_hit[c]:.2f}",
                                     "MRR of the exact page": f"{r.gold_chunk_mrr[c]:.2f}"} for c in order])), ""]

    cost = main[main.model.isin(models)].groupby("config")[["mean_prompt_tokens", "median_latency_s"]].mean()
    out += ["## Cost of retrieval, mean over models", "",
            table(pd.DataFrame([{"config": RETRIEVER_LABEL[c],
                                 "prompt tokens per question": f"{cost.mean_prompt_tokens[c]:.0f}",
                                 "median seconds per answer": f"{cost.median_latency_s[c]:.1f}"}
                                for c in MAIN])), "",
            f"Unparsed answers (counted as wrong): {int(df.unparsed.sum())} of {len(df)}.", ""]
    path.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description="Score benchmark answers and draw the figures.")
    p.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    p.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    p.add_argument("--sweep-model", default="deepseek-flash",
                   help="model used for the top-k and source-ablation figures")
    args = p.parse_args()

    df, retrieval = load(args.bench_dir)
    if df.empty:
        raise SystemExit(f"no answers found in {args.bench_dir}")
    summary, tests = summarise(df), paired_tests(df)
    totals = overall(summary)

    # Outcomes only: the answer letters of graded quizzes stay out of the committed file.
    df.drop(columns=["gold", "pred"]).to_csv(args.bench_dir / "per_question.csv", index=False)
    summary.round(4).to_csv(args.bench_dir / "summary.csv", index=False)
    totals.round(4).to_csv(args.bench_dir / "overall.csv", index=False)
    tests.round(4).to_csv(args.bench_dir / "paired_tests.csv", index=False)
    if not retrieval.empty:
        (retrieval.drop(columns="qid").groupby(["config", "dataset"]).mean().round(4)
         .to_csv(args.bench_dir / "retrieval.csv"))

    write_markdown(df, summary, totals, tests, retrieval, args.bench_dir / "RESULTS.md")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_rag_effect(summary, args.fig_dir / "01_rag_effect_by_model.png")
    fig_retrievers(summary, args.fig_dir / "02_retriever_comparison.png")
    fig_flips(tests, args.fig_dir / "03_helped_vs_hurt.png")
    fig_multilabel(summary, args.fig_dir / "04_multilabel_metrics.png")
    fig_topk(summary, args.fig_dir / "05_topk_sensitivity.png", args.sweep_model)
    fig_sources(summary, args.fig_dir / "06_source_ablation.png", args.sweep_model)
    if not retrieval.empty:
        fig_retrieval(retrieval, args.fig_dir / "07_retrieval_quality.png")

    pd.set_option("display.width", 200)
    table = totals[totals.config.isin(MAIN)].pivot_table(index="model", columns="config", values="accuracy")
    print("pooled exact-match accuracy over all questions")
    print((table * 100).round(1).to_string())
    print(f"\nunparsed answers: {int(df.unparsed.sum())} of {len(df)}")
    print(f"tables -> {args.bench_dir}\nfigures -> {args.fig_dir}")


if __name__ == "__main__":
    main()
