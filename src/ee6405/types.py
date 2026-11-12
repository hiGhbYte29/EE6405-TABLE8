"""Shared data types.

Every module exchanges these three objects, so the P0/P1/P2/P3 workstreams can
be built in parallel against a stable contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """One source file after loading, before chunking."""

    doc_id: str
    text: str
    source: str  # path or URL it came from
    modality: str  # "slides" | "code" | "transcript" | "doc" | "quiz" | "markdown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit of text."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    modality: str
    position: int = 0  # ordinal of the chunk inside its document
    metadata: dict[str, Any] = field(default_factory=dict)
    locator: str = ""
    week: str | None = None


@dataclass
class Hit:
    """A chunk returned by a retriever, with its score."""

    chunk: Chunk
    score: float
    retriever: str = ""  # "bm25" | "dense" | "hybrid"

    def citation(self) -> str:
        return f"{self.chunk.source}#{self.chunk.position}"
