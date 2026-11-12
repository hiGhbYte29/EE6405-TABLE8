"""Configuration, read from the environment (``.env`` via python-dotenv).

Copy ``.env.example`` to ``.env`` and fill it in. Nothing here holds a default
API key, and ``.env`` is git-ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional at import time
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")


def _path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    llm_base_url: str = os.getenv("EE6405_LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = os.getenv("EE6405_LLM_API_KEY", "")
    llm_model: str = os.getenv("EE6405_LLM_MODEL", "gpt-4o-mini")
    # DeepSeek thinking mode: "enabled" | "disabled" | "" (don't send; provider default)
    llm_thinking: str = os.getenv("EE6405_LLM_THINKING", "")

    # --- embeddings ---
    embed_backend: str = os.getenv("EE6405_EMBED_BACKEND", "local")
    embed_model: str = os.getenv(
        "EE6405_EMBED_MODEL", "BAAI/bge-m3"
    )
    embed_api_key: str = os.getenv("EE6405_EMBED_API_KEY", "")
    embed_base_url: str = os.getenv("EE6405_EMBED_BASE_URL", "")

    # --- retrieval ---
    top_k: int = int(os.getenv("EE6405_TOP_K", "5"))
    chunk_size: int = int(os.getenv("EE6405_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("EE6405_CHUNK_OVERLAP", "120"))
    retriever: str = os.getenv("EE6405_RETRIEVER", "hybrid")

    # --- paths ---
    data_dir: Path = _path(os.getenv("EE6405_DATA_DIR", "data"))
    index_path: Path = _path(os.getenv("EE6405_INDEX_PATH", "data/processed/index.sqlite"))
    memory_path: Path = _path(os.getenv("EE6405_MEMORY_PATH", "data/processed/memory.sqlite"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def require_llm_key(self) -> str:
        if not self.llm_api_key:
            raise RuntimeError(
                "EE6405_LLM_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.llm_api_key


settings = Settings()
