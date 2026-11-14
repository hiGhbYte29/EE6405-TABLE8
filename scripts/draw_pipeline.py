"""Draw the system diagram used in the slides.

    python scripts/draw_pipeline.py            # writes reports/figures/00_system_pipeline.png

Two lanes: building the index once (top), answering a question (bottom).
Counts are read from the index, so the picture cannot drift from the data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK, SECONDARY, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE, BLUE_BG, LINE = "#2a78d6", "#e8f1fc", "#c3c2b7"
GREEN_BG, GREEN = "#e3f5ee", "#1baf7a"
NEUTRAL_BG = "#f1f0ec"


def counts() -> dict[str, int]:
    conn = sqlite3.connect(ROOT / "data" / "processed" / "index.sqlite")
    rows = dict(conn.execute("SELECT modality, COUNT(*) FROM chunks GROUP BY modality"))
    files = dict(conn.execute("SELECT modality, COUNT(DISTINCT source) FROM chunks GROUP BY modality"))
    conn.close()
    return {"chunks": sum(rows.values()), **{f"n_{k}": v for k, v in rows.items()},
            **{f"f_{k}": v for k, v in files.items()}}


def box(ax, x, y, w, h, title, lines=(), face=NEUTRAL_BG, edge=LINE, title_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12",
                                fc=face, ec=edge, lw=1.1, zorder=2))
    top = y + h - 0.26
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=11.5, fontweight="bold",
            color=title_color, zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - 0.36 - i * 0.29, line, ha="center", va="top", fontsize=9.6,
                color=SECONDARY, zorder=3)


def arrow(ax, a, b, color=MUTED, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=2, shrinkB=2))


def main() -> None:
    c = counts()
    plt.rcParams.update({"font.family": ["Segoe UI", "Microsoft YaHei", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(15, 7.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    ax.text(0.15, 7.15, "EE6405 study assistant: how a question becomes a cited answer",
            fontsize=15.5, fontweight="bold", color=INK, va="top")

    # ---- lane 1: build the index once
    ax.text(0.15, 6.42, "BUILD THE INDEX  (offline, once)", fontsize=10, color=MUTED, fontweight="bold")
    h = 1.72
    y1 = 4.45
    box(ax, 0.15, y1, 3.2, h, "Course material",
        [f"{c.get('f_slides', 0)} slide decks  ·  {c.get('f_transcript', 0)} lecture videos",
         f"{c.get('f_notebook', 0)} code notebooks",
         "quizzes, syllabus, clarifications"])
    box(ax, 3.85, y1, 3.3, h, "Extract text",
        ["PDF text per page", "Whisper large-v3 + term fixes (video)",
         "vision-LLM OCR (scanned quizzes)"])
    box(ax, 7.65, y1, 3.0, h, "Chunk by structure",
        ["one slide page  ·  one notebook cell", "~60 s of speech with timestamps",
         "keeps week + page / cell / time"])
    box(ax, 11.15, y1, 3.7, h, f"Index  ·  {c['chunks']:,} chunks",
        ["SQLite FTS5  →  BM25 (exact terms)", "BGE-m3 vectors, 1024-d  →  dense",
         "multilingual: Chinese question, English slide"], face=BLUE_BG, edge=BLUE, title_color="#184f95")
    for x0, x1 in [(3.35, 3.85), (7.15, 7.65), (10.65, 11.15)]:
        arrow(ax, (x0, y1 + h / 2), (x1, y1 + h / 2))

    # ---- lane 2: answer a question
    ax.text(0.15, 3.72, "ANSWER A QUESTION  (online, ~2 s)", fontsize=10, color=MUTED, fontweight="bold")
    y2 = 1.75
    box(ax, 0.15, y2, 2.3, h, "Question", ["English or Chinese", "optional filters:", "week, material type"])
    box(ax, 2.95, y2, 3.3, h, "Retrieve top-k",
        ["BM25  |  dense  |  hybrid", "hybrid = reciprocal rank fusion",
         "of the two rankings"], face=BLUE_BG, edge=BLUE, title_color="#184f95")
    box(ax, 6.75, y2, 2.9, h, "Build the prompt",
        ["numbered context blocks [1]..[k]", "+ recent turns (memory)",
         "rule: answer from the context"])
    box(ax, 10.15, y2, 2.0, h, "LLM", ["DeepSeek", "(any OpenAI-", "compatible API)"])
    box(ax, 12.65, y2, 2.2, h, "Answer", ["with citations", "[1] [2] ...", "to the chunks used"],
        face=GREEN_BG, edge=GREEN, title_color="#0b6b4a")
    for x0, x1 in [(2.45, 2.95), (6.25, 6.75), (9.65, 10.15), (12.15, 12.65)]:
        arrow(ax, (x0, y2 + h / 2), (x1, y2 + h / 2))
    # the index feeds retrieval
    arrow(ax, (13.0, y1), (4.6, y2 + h), color=BLUE, rad=0.0)
    ax.text(8.75, 3.98, "search", fontsize=9.5, color="#184f95", ha="center", va="center",
            bbox={"fc": SURFACE, "ec": "none", "pad": 1.5}, zorder=4)

    # ---- what the student sees
    ax.text(0.15, 1.2, "WHAT THE STUDENT SEES", fontsize=10, color=MUTED, fontweight="bold")
    items = [
        ("Source panel", "click a citation → the PDF page, the video at that second, the notebook cell"),
        ("Course knowledge map", "materials linked through the concepts they share; lights up what the answer used"),
    ]
    for i, (name, text) in enumerate(items):
        ax.text(0.15, 0.78 - i * 0.4, name, fontsize=10.8, fontweight="bold", color=INK, va="center")
        ax.text(2.75, 0.78 - i * 0.4, text, fontsize=10.2, color=SECONDARY, va="center")

    out = ROOT / "reports" / "figures" / "00_system_pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
