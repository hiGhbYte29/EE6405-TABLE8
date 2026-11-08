"""Retrievers over a :class:`~ee6405.rag.store.ChunkStore`.

Three strategies, all returning a ranked ``list[Hit]``:

``BM25Retriever``   SQLite FTS5 lexical match -- strong on exact jargon
``DenseRetriever``  cosine similarity over embeddings -- strong on paraphrase
``HybridRetriever`` reciprocal rank fusion of the two

Swapping the retriever is the main *without-RAG-variant* ablation in the
report, so keep the interface stable.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ee6405.config import settings
from ee6405.rag.embedding import Embedder, build_embedder
from ee6405.rag.store import ChunkStore
from ee6405.types import Hit


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, top_k: int | None = None, *, week: str | int | None = None, modality: str | list[str] | None = None, exclude_modality: str | list[str] | None = None) -> list[Hit]:
        ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, store: ChunkStore) -> None:
        self.store = store

    def retrieve(self, query: str, top_k: int | None = None, *, week: str | int | None = None, modality: str | list[str] | None = None, exclude_modality: str | list[str] | None = None) -> list[Hit]:
        k = top_k or settings.top_k
        return [
            Hit(chunk=chunk, score=score, retriever=self.name)
            for chunk, score in self.store.search_bm25(query, k, week=week, modality=modality, exclude_modality=exclude_modality)
        ]


class DenseRetriever:
    name = "dense"

    def __init__(self, store: ChunkStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or build_embedder()
        self._ids, self._matrix = store.load_embedding_matrix()

    def retrieve(self, query: str, top_k: int | None = None, *, week: str | int | None = None, modality: str | list[str] | None = None, exclude_modality: str | list[str] | None = None) -> list[Hit]:
        k = top_k or settings.top_k
        if self._matrix.size == 0:
            return []  # index built without embeddings
        allowed = []
        for i, cid in enumerate(self._ids):
            c = self.store.get(cid)
            if c is None: continue
            if week is not None and str(c.week) != (f"{int(week):02d}" if str(week).isdigit() else str(week)): continue
            if modality is not None and c.modality not in ([modality] if isinstance(modality, str) else modality): continue
            if exclude_modality is not None and c.modality in ([exclude_modality] if isinstance(exclude_modality, str) else exclude_modality): continue
            allowed.append(i)
        if not allowed: return []
        qvec = self.embedder.encode([query])[0]
        if qvec.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"embedding dim mismatch: query {qvec.shape[0]} vs index "
                f"{self._matrix.shape[1]}. Re-run the ingest pipeline with this embedder."
            )
        scores = self._matrix @ qvec
        order = np.array(allowed)[np.argsort(-scores[allowed])[:k]]
        hits: list[Hit] = []
        for idx in order:
            chunk = self.store.get(self._ids[int(idx)])
            if chunk is not None:
                hits.append(Hit(chunk=chunk, score=float(scores[idx]), retriever=self.name))
        return hits


class HybridRetriever:
    """Reciprocal rank fusion: ``score = sum over retrievers of 1 / (k0 + rank)``.

    RRF needs no score calibration between BM25 and cosine, which is why it is
    the default here.
    """

    name = "hybrid"

    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder | None = None,
        k0: int = 60,
        pool_factor: int = 4,
    ) -> None:
        self.bm25 = BM25Retriever(store)
        self.dense = DenseRetriever(store, embedder)
        self.k0 = k0
        self.pool_factor = pool_factor

    def retrieve(self, query: str, top_k: int | None = None, *, week: str | int | None = None, modality: str | list[str] | None = None, exclude_modality: str | list[str] | None = None) -> list[Hit]:
        k = top_k or settings.top_k
        pool = k * self.pool_factor

        fused: dict[str, float] = {}
        chunks = {}
        for retriever in (self.bm25, self.dense):
            for rank, hit in enumerate(retriever.retrieve(query, pool, week=week, modality=modality, exclude_modality=exclude_modality), start=1):
                cid = hit.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.k0 + rank)
                chunks[cid] = hit.chunk

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            Hit(chunk=chunks[cid], score=score, retriever=self.name) for cid, score in ranked
        ]


def build_retriever(
    store: ChunkStore,
    kind: str | None = None,
    embedder: Embedder | None = None,
) -> Retriever:
    """Factory. ``kind`` defaults to ``EE6405_RETRIEVER``."""
    kind = (kind or settings.retriever).lower()
    if kind == "bm25":
        return BM25Retriever(store)
    if kind == "dense":
        return DenseRetriever(store, embedder)
    if kind == "hybrid":
        return HybridRetriever(store, embedder)
    raise ValueError(f"unknown retriever: {kind!r} (expected bm25 | dense | hybrid)")
