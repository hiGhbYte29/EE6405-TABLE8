"""LLM access behind one small interface.

Everything talks to an OpenAI-*compatible* endpoint, so the same code drives
OpenAI, DeepSeek, Qwen/DashScope-compatible, SiliconFlow, vLLM or Ollama just
by changing ``EE6405_LLM_BASE_URL`` / ``EE6405_LLM_MODEL`` in ``.env``.
That is what makes the baseline comparison in the report cheap to run.

:class:`EchoLLM` is an offline stand-in: it lets the pipeline, the Streamlit
app and the tests run with no API key and no network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from ee6405.config import settings

log = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "user"|"system"|"assistant", "content": str}


class LLM(Protocol):
    name: str

    def chat(self, messages: list[Message], temperature: float = 0.2) -> str:
        ...


@dataclass
class EchoLLM:
    """Offline stub. Returns the retrieved context verbatim.

    Not a baseline -- it exists so that a teammate without a key can still
    exercise ingest -> retrieve -> prompt -> UI end to end.
    """

    name: str = "echo"
    calls: list[list[Message]] = field(default_factory=list)

    def chat(self, messages: list[Message], temperature: float = 0.2) -> str:
        self.calls.append(messages)
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return (
            "[EchoLLM -- no API key configured, so no model was called]\n\n"
            "The prompt that would have been sent ends with:\n\n"
            f"{user[-1200:]}"
        )


class LLMClient:
    """Thin wrapper over the OpenAI SDK pointed at any compatible endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        self.name = model or settings.llm_model
        self._client = OpenAI(
            api_key=api_key or settings.require_llm_key(),
            base_url=base_url or settings.llm_base_url,
            timeout=timeout,
        )

    def chat(self, messages: list[Message], temperature: float = 0.2) -> str:
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()


def build_llm(model: str | None = None, allow_offline: bool = True) -> LLM:
    """Return a real client, or :class:`EchoLLM` when no key is configured."""
    if not settings.llm_api_key:
        if not allow_offline:
            raise RuntimeError("EE6405_LLM_API_KEY is not set.")
        log.warning("EE6405_LLM_API_KEY is not set -- using EchoLLM (offline stub).")
        return EchoLLM()
    return LLMClient(model=model)
