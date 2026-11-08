"""SQLite chunk store with FTS5 and optional dense vectors."""
from __future__ import annotations
import json, sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
import numpy as np
from ee6405.types import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
 chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, text TEXT NOT NULL, source TEXT NOT NULL,
 modality TEXT NOT NULL, position INTEGER NOT NULL, metadata TEXT NOT NULL DEFAULT '{}',
 week TEXT, locator TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_modality ON chunks(modality);
CREATE INDEX IF NOT EXISTS idx_chunks_week ON chunks(week);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS embeddings (chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE, dim INTEGER NOT NULL, vector BLOB NOT NULL);
"""
_FTS_SPECIALS = str.maketrans({c: " " for c in '"*()[]{}^:-+~'})

class ChunkStore:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path=Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn=sqlite3.connect(self.path); self.conn.row_factory=sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if not read_only:
            self.conn.executescript(_SCHEMA)
            # Upgrade indexes created by older versions in place.
            cols={r[1] for r in self.conn.execute("PRAGMA table_info(chunks)")}
            if "week" not in cols: self.conn.execute("ALTER TABLE chunks ADD COLUMN week TEXT")
            if "locator" not in cols: self.conn.execute("ALTER TABLE chunks ADD COLUMN locator TEXT NOT NULL DEFAULT ''")
            self.conn.commit()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
    def close(self): self.conn.close()
    def reset(self): self.conn.executescript("DELETE FROM embeddings; DELETE FROM chunks_fts; DELETE FROM chunks;"); self.conn.commit()
    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        rows=[(c.chunk_id,c.doc_id,c.text,c.source,c.modality,c.position,json.dumps(c.metadata,ensure_ascii=False),c.week or c.metadata.get("week"),c.locator or c.metadata.get("locator", "")) for c in chunks]
        if not rows:return 0
        self.conn.executemany("INSERT OR REPLACE INTO chunks (chunk_id,doc_id,text,source,modality,position,metadata,week,locator) VALUES (?,?,?,?,?,?,?,?,?)",rows)
        self.conn.executemany("DELETE FROM chunks_fts WHERE chunk_id=?",[(r[0],) for r in rows])
        self.conn.executemany("INSERT INTO chunks_fts (chunk_id,text) VALUES (?,?)",[(r[0],r[2]) for r in rows]); self.conn.commit(); return len(rows)
    def add_embeddings(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> int:
        if len(chunk_ids)!=len(vectors): raise ValueError("chunk_ids and vectors must have the same length")
        vec=np.asarray(vectors,dtype=np.float32); self.conn.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?,?)",[(cid,int(v.shape[0]),v.tobytes()) for cid,v in zip(chunk_ids,vec)]); self.conn.commit(); return len(chunk_ids)
    @staticmethod
    def _to_chunk(r):
        return Chunk(r["chunk_id"],r["doc_id"],r["text"],r["source"],r["modality"],r["position"],json.loads(r["metadata"]),r["locator"] or "",r["week"])
    def count(self): return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    def count_embeddings(self): return int(self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    def stats(self): return {r["modality"]:r["n"] for r in self.conn.execute("SELECT modality,COUNT(*) n FROM chunks GROUP BY modality ORDER BY n DESC")}
    def all_chunks(self): return [self._to_chunk(r) for r in self.conn.execute("SELECT * FROM chunks ORDER BY doc_id,position")]
    def get(self, chunk_id):
        r=self.conn.execute("SELECT * FROM chunks WHERE chunk_id=?",(chunk_id,)).fetchone(); return self._to_chunk(r) if r else None
    def load_embedding_matrix(self):
        rows=self.conn.execute("SELECT chunk_id,vector FROM embeddings ORDER BY chunk_id").fetchall()
        if not rows:return [],np.zeros((0,0),dtype=np.float32)
        ids=[r["chunk_id"] for r in rows]; mat=np.vstack([np.frombuffer(r["vector"],dtype=np.float32) for r in rows]); return ids,mat/np.clip(np.linalg.norm(mat,axis=1,keepdims=True),1e-12,None)
    def search_bm25(self, query: str, top_k: int=5, *, week: str|int|None=None, modality: str|Sequence[str]|None=None, exclude_modality: str|Sequence[str]|None=None):
        cleaned=query.translate(_FTS_SPECIALS).strip()
        if not cleaned:return []
        # Natural-language questions should match any informative term; AND
        # matching makes BM25 return nothing when the full question is not in a chunk.
        cleaned = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in cleaned.split() if t)

        clauses=["chunks_fts MATCH ?"]; args=[cleaned]
        if week is not None: clauses.append("c.week=?"); args.append(f"{int(week):02d}" if str(week).isdigit() else str(week))
        if modality is not None:
            mods=[modality] if isinstance(modality,str) else list(modality); clauses.append("c.modality IN ("+",".join("?"*len(mods))+")"); args.extend(mods)
        if exclude_modality is not None:
            excluded=[exclude_modality] if isinstance(exclude_modality,str) else list(exclude_modality); clauses.append("c.modality NOT IN ("+",".join("?"*len(excluded))+")"); args.extend(excluded)
        sql="SELECT c.*,bm25(chunks_fts) rank FROM chunks_fts JOIN chunks c ON c.chunk_id=chunks_fts.chunk_id WHERE " + " AND ".join(clauses) + " ORDER BY rank LIMIT ?"; args.append(top_k)
        try: rows=self.conn.execute(sql,args).fetchall()
        except sqlite3.OperationalError:return []
        return [(self._to_chunk(r),-float(r["rank"])) for r in rows]
