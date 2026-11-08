"""Build the RAG index: ``data/raw/`` -> ``data/processed/index.sqlite``.

    python -m ee6405.ingest.pipeline --source data/raw --rebuild
    python -m ee6405.ingest.pipeline --no-embed          # lexical index only, fast
    python -m ee6405.ingest.pipeline --chunk-size 400 --chunk-overlap 80
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ee6405.config import settings
from ee6405.ingest.chunking import chunk_document
from ee6405.ingest.loaders import load_tree
from ee6405.rag.embedding import build_embedder
from ee6405.rag.store import ChunkStore
from ee6405.types import Chunk

log = logging.getLogger("ee6405.ingest")


def build_index(
    source: Path,
    index_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool = False,
    embed: bool = True,
    embed_backend: str | None = None,
    modalities: set[str] | None = None,
) -> dict[str, int]:
    store = ChunkStore(index_path)
    if rebuild:
        store.reset()

    n_docs = 0
    all_chunks: list[Chunk] = []
    for doc in load_tree(source, modalities=modalities):
        n_docs += 1
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        log.info("%-60s %-10s %4d chunks", doc.metadata.get("filename", doc.source),
                 doc.modality, len(chunks))

    store.add_chunks(all_chunks)

    n_embedded = 0
    if embed and all_chunks:
        embedder = build_embedder(embed_backend)
        log.info("embedding %d chunks with %s ...", len(all_chunks), embedder.name)
        vectors = embedder.encode([c.text for c in all_chunks])
        n_embedded = store.add_embeddings([c.chunk_id for c in all_chunks], vectors)

    summary = {
        "documents": n_docs,
        "chunks": len(all_chunks),
        "embedded": n_embedded,
        "total_in_index": store.count(),
        "by_modality": store.stats(),
    }
    log.info("by modality: %s", store.stats())
    store.close()
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Build the EE6405 RAG index.")
    p.add_argument("--source", type=Path, default=settings.raw_dir,
                   help="directory of course material (default: data/raw)")
    p.add_argument("--index", type=Path, default=settings.index_path)
    p.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    p.add_argument("--rebuild", action="store_true", help="wipe the index first")
    p.add_argument("--no-embed", dest="embed", action="store_false",
                   help="skip embeddings (BM25-only index, much faster)")
    p.add_argument("--embed-backend", default=None, choices=["local", "bge-m3", "api", "hashing"])
    p.add_argument("--modality", action="append", choices=["slides", "clarification", "notebook", "mcq", "assessment_pdf", "misc"], help="build only selected modality (repeatable)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if not args.source.exists() or not any(args.source.rglob("*")):
        raise SystemExit(
            f"No course material found in {args.source}.\n"
            "Put lecture slides / code / transcripts / docs / quizzes under data/raw/ first.\n"
            "See docs/DATA.md for the expected layout."
        )

    summary = build_index(
        source=args.source,
        index_path=args.index,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild,
        embed=args.embed,
        embed_backend=args.embed_backend,
        modalities=set(args.modality) if args.modality else None,
    )
    print(f"index: {args.index}")
    for key, value in summary.items():
        if key == "by_modality":
            print("  by_modality:")
            for modality, count in value.items():
                print(f"    {modality:16s} {count}")
        else:
            print(f"  {key:16s} {value}")


if __name__ == "__main__":
    main()
