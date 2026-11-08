"""Embedding backends.

Three interchangeable implementations behind one ``encode()`` call, so the
report can compare them as a hyperparameter:

``local``   sentence-transformers, runs offline (needs requirements-ml.txt)
``api``     any OpenAI-compatible ``/embeddings`` endpoint
``hashing`` dependency-free hashing trick -- a weak but honest TF baseline,
            and what keeps tests and CI runnable without torch
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from ee6405.config import settings

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array."""
        ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


class HashingEmbedder:
    """Hashing-trick bag of words. No model download, deterministic.

    Useful as the "no semantic embedding" ablation and as a test fixture.
    """

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        return _normalise(out)


class LocalEmbedder:
    """sentence-transformers backend (``pip install -r requirements-ml.txt``)."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name or settings.embed_model
        self._model = SentenceTransformer(self.name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return _normalise(vecs)


class ApiEmbedder:
    """OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(self, model_name: str | None = None) -> None:
        from openai import OpenAI

        self.name = model_name or settings.embed_model
        self._client = OpenAI(
            api_key=settings.embed_api_key or settings.llm_api_key,
            base_url=settings.embed_base_url or settings.llm_base_url,
        )
        self.dim = 0  # discovered on first call

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            resp = self._client.embeddings.create(
                model=self.name, input=texts[start : start + 64]
            )
            vectors.extend(item.embedding for item in resp.data)
        mat = _normalise(np.asarray(vectors, dtype=np.float32))
        self.dim = int(mat.shape[1]) if mat.size else 0
        return mat


class BGEM3Embedder:
    """Local BAAI/bge-m3 sentence-transformers backend with safe GPU fallback."""
    def __init__(self, model_name: str | None = None, batch_size: int = 8) -> None:
        from sentence_transformers import SentenceTransformer
        self.name = model_name or "BAAI/bge-m3"
        self.batch_size = batch_size
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        try:
            self._model = SentenceTransformer(self.name, device=device)
        except (RuntimeError, ValueError):
            self._model = SentenceTransformer(self.name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())
    def encode(self, texts: list[str]) -> np.ndarray:
        try:
            vecs = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        except RuntimeError:
            self._model.to("cpu")
            vecs = self._model.encode(texts, batch_size=4, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return _normalise(vecs)


def build_embedder(backend: str | None = None) -> Embedder:
    """Factory. ``backend`` defaults to ``EE6405_EMBED_BACKEND``.

    Falls back to :class:`HashingEmbedder` with a warning rather than crashing,
    so a teammate without torch installed can still run the pipeline.
    """
    backend = (backend or settings.embed_backend).lower()
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "api":
        return ApiEmbedder()
    if backend in {"local", "bge-m3", "bge_m3"}:
        try:
            return BGEM3Embedder()
        except ImportError:
            import warnings

            warnings.warn(
                "sentence-transformers is not installed or BGE could not be loaded; falling back to HashingEmbedder. "
                "Run `make setup-ml` for real embeddings.",
                stacklevel=2,
            )
            return HashingEmbedder()
    raise ValueError(f"unknown embedding backend: {backend!r}")
