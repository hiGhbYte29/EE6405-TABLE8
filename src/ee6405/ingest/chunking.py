"""Modality-aware chunking. Loaders already emit retrieval-sized logical units."""
from __future__ import annotations
from ee6405.types import Chunk, Document

def chunk_document(doc: Document, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Chunk]:
    if not doc.text.strip(): return []
    # Preserve page/question/cell boundaries for every real dataset modality.
    return [Chunk(chunk_id=f"{doc.doc_id}-0000", doc_id=doc.doc_id, text=doc.text.strip(),
        source=doc.source, modality=doc.modality, position=int(doc.metadata.get("page", doc.metadata.get("cell_idx", 0)) or 0),
        metadata=dict(doc.metadata), locator=str(doc.metadata.get("locator", "")), week=doc.metadata.get("week"))]
