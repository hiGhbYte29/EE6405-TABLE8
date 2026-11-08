"""Conversation memory -- the component the *without-memory* ablation removes.

Two layers, both in one SQLite file:

``turns``  short-term: the verbatim recent exchange, replayed into the prompt
``facts``  long-term: durable statements about the learner ("exam is week 13",
           "struggles with CRF"), recalled by lexical overlap with the question

Keeping long-term memory as explicit rows rather than a rolling summary makes
the ablation clean and the behaviour explainable in the demo video.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    fact       TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(session_id, fact)
);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);
"""

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


@dataclass
class Turn:
    role: str
    content: str


class ConversationMemory:
    def __init__(self, path: str | Path, session_id: str = "default") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def __enter__(self) -> ConversationMemory:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # -- short term --------------------------------------------------------- #
    def add_turn(self, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (self.session_id, role, content, time.time()),
        )
        self.conn.commit()

    def recent_turns(self, limit: int = 6) -> list[Turn]:
        rows = self.conn.execute(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (self.session_id, limit),
        ).fetchall()
        return [Turn(r["role"], r["content"]) for r in reversed(rows)]

    # -- long term ---------------------------------------------------------- #
    def remember(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        self.conn.execute(
            "INSERT OR IGNORE INTO facts (session_id, fact, created_at) VALUES (?, ?, ?)",
            (self.session_id, fact, time.time()),
        )
        self.conn.commit()

    def all_facts(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT fact FROM facts WHERE session_id = ? ORDER BY id", (self.session_id,)
        ).fetchall()
        return [r["fact"] for r in rows]

    def recall(self, query: str, limit: int = 5) -> list[str]:
        """Token-overlap recall.

        TODO(P1): swap for embedding similarity once the dense index is stable;
        keep this version as the cheap baseline for the memory ablation table.
        """
        q = set(_TOKEN.findall(query.lower()))
        if not q:
            return []
        scored = []
        for fact in self.all_facts():
            f = set(_TOKEN.findall(fact.lower()))
            if f:
                scored.append((len(q & f) / len(q | f), fact))
        scored.sort(key=lambda kv: -kv[0])
        return [fact for score, fact in scored[:limit] if score > 0]

    def clear(self) -> None:
        self.conn.execute("DELETE FROM turns WHERE session_id = ?", (self.session_id,))
        self.conn.execute("DELETE FROM facts WHERE session_id = ?", (self.session_id,))
        self.conn.commit()
