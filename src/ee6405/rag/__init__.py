"""P1 -- the RAG database and retrievers.

``store``      SQLite persistence (chunks + FTS5 lexical index + embedding blobs)
``embedding``  pluggable embedding backends
``retriever``  bm25 / dense / hybrid retrieval over the store
"""

from ee6405.rag.retriever import build_retriever
from ee6405.rag.store import ChunkStore

__all__ = ["ChunkStore", "build_retriever"]
