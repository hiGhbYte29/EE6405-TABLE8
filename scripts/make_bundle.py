"""Package the prebuilt index and transcripts so teammates can skip the build.

The index and the course material are not in git (too large, and the index is a
build artefact). This produces the zip referenced in README section 3:

    python scripts/make_bundle.py --out dist/EE6405_RAG_Bundle.zip

Unzipping it at the repo root puts every file where the code expects it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (source on disk, path inside the zip)
CONTENTS = [
    ("data/processed/index.sqlite", "data/processed/index.sqlite"),
    ("data/interim/transcripts_corrected", "data/interim/transcripts_corrected"),
    ("data/interim/coding_quiz_ocr", "data/interim/coding_quiz_ocr"),
]


def index_summary(path: Path) -> dict:
    conn = sqlite3.connect(path)
    try:
        modalities = dict(
            conn.execute(
                "SELECT modality, COUNT(*) FROM chunks GROUP BY modality ORDER BY 2 DESC"
            ).fetchall()
        )
        return {
            "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "embeddings": conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0],
            "embedding_dim": conn.execute(
                "SELECT DISTINCT dim FROM embeddings"
            ).fetchone()[0],
            "modalities": modalities,
        }
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the RAG deployment bundle.")
    ap.add_argument("--out", type=Path, default=REPO / "dist" / "EE6405_RAG_Bundle.zip")
    args = ap.parse_args()

    missing = [src for src, _ in CONTENTS if not (REPO / src).exists()]
    if missing:
        raise SystemExit("missing, build the index first:\n  " + "\n  ".join(missing))

    args.out.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "bundle": args.out.name,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index": index_summary(REPO / "data/processed/index.sqlite"),
        "note": "Unzip at the repository root. See README section 3.",
        "files": [],
    }

    staged: list[tuple[Path, str]] = []
    for src, dest in CONTENTS:
        node = REPO / src
        if node.is_file():
            staged.append((node, dest))
        else:
            for f in sorted(node.rglob("*")):
                if f.is_file():
                    staged.append((f, f"{dest}/{f.relative_to(node).as_posix()}"))

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path, arc in staged:
            z.write(path, arc)
            manifest["files"].append({"path": arc, "bytes": path.stat().st_size})
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    size_mb = args.out.stat().st_size / 1e6

    print(f"{args.out}")
    print(f"  {len(staged)} files, {size_mb:.1f} MB")
    print(f"  sha256 {digest}")
    print(f"  index  {manifest['index']['chunks']} chunks, "
          f"{manifest['index']['embeddings']} embeddings")
    for modality, n in manifest["index"]["modalities"].items():
        print(f"    {modality:16s} {n}")


if __name__ == "__main__":
    main()
