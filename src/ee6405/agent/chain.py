"""The reasoning chain: question -> (memory + retrieval) -> grounded answer.

``StudyAssistant.answer()`` carries two switches, ``use_rag`` and
``use_memory``, which are exactly the three rows of the ablation table in
``Ideas/学习助手.md``::

    base            use_rag=True,  use_memory=True
    without rag     use_rag=False, use_memory=True
    without memory  use_rag=True,  use_memory=False
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ee6405.agent.llm import LLM, Message, build_llm
from ee6405.agent.memory import ConversationMemory
from ee6405.config import settings
from ee6405.rag.retriever import Retriever
from ee6405.types import Hit

SYSTEM_PROMPT = """You are a study assistant for NTU course EE6405 (Natural Language Processing).

Rules:
- Answer only from the COURSE CONTEXT below when it is relevant.
- Cite the sources you used as [1], [2], ... matching the numbered context blocks.
- If the context does not contain the answer, say so plainly, then answer from
  general NLP knowledge and label that part "(outside course material)".
- Be concise and concrete. Prefer the course's own terminology and notation.
"""


@dataclass
class Answer:
    text: str
    hits: list[Hit] = field(default_factory=list)
    used_memory: list[str] = field(default_factory=list)
    prompt: list[Message] = field(default_factory=list)
    model: str = ""

    @property
    def citations(self) -> list[str]:
        return [h.citation() for h in self.hits]


def format_context(hits: list[Hit]) -> str:
    """Render hits as numbered blocks the model can cite as [1], [2], ..."""
    if not hits:
        return "(no course material retrieved)"
    return "\n\n".join(
        f"[{i}] source={h.chunk.source} modality={h.chunk.modality}\n{h.chunk.text}"
        for i, h in enumerate(hits, start=1)
    )


class StudyAssistant:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLM | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or build_llm()
        self.memory = memory

    def build_prompt(
        self, question: str, hits: list[Hit], facts: list[str], history: list[Message]
    ) -> list[Message]:
        parts = []
        if facts:
            parts.append("WHAT YOU REMEMBER ABOUT THIS LEARNER:\n" + "\n".join(f"- {f}" for f in facts))
        parts.append("COURSE CONTEXT:\n" + format_context(hits))
        parts.append(f"QUESTION:\n{question}")

        messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": "\n\n".join(parts)})
        return messages

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        use_rag: bool = True,
        use_memory: bool = True,
        temperature: float = 0.2,
    ) -> Answer:
        k = top_k or settings.top_k

        hits: list[Hit] = []
        if use_rag and self.retriever is not None:
            hits = self.retriever.retrieve(question, k)

        facts: list[str] = []
        history: list[Message] = []
        if use_memory and self.memory is not None:
            facts = self.memory.recall(question)
            history = [
                {"role": t.role, "content": t.content} for t in self.memory.recent_turns(6)
            ]

        prompt = self.build_prompt(question, hits, facts, history)
        text = self.llm.chat(prompt, temperature=temperature)

        if use_memory and self.memory is not None:
            self.memory.add_turn("user", question)
            self.memory.add_turn("assistant", text)

        return Answer(
            text=text,
            hits=hits,
            used_memory=facts,
            prompt=prompt,
            model=getattr(self.llm, "name", "unknown"),
        )
