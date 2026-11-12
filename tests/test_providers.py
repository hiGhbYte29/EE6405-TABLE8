import pytest

from ee6405.agent.llm import EchoLLM, LLMClient
from ee6405.eval.run_eval import MissingKeyError, build_provider_llm, evaluate_mcq
from ee6405.rag.store import ChunkStore
from ee6405.types import Chunk

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "TEST_DEEPSEEK_KEY",
        "model": "deepseek-chat",
    }
}


def test_provider_uses_its_own_endpoint_key_and_model(monkeypatch):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "sk-test")
    llm = build_provider_llm("deepseek", PROVIDERS)
    assert isinstance(llm, LLMClient)
    assert llm.base_url == "https://api.deepseek.com"
    assert llm.name == "deepseek-chat"


def test_model_override_beats_provider_default(monkeypatch):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "sk-test")
    assert build_provider_llm("deepseek", PROVIDERS, model="deepseek-reasoner").name == "deepseek-reasoner"


def test_missing_provider_key_fails_instead_of_echoing(monkeypatch):
    monkeypatch.delenv("TEST_DEEPSEEK_KEY", raising=False)
    with pytest.raises(MissingKeyError, match="TEST_DEEPSEEK_KEY"):
        build_provider_llm("deepseek", PROVIDERS)


def test_unknown_provider_is_rejected():
    with pytest.raises(SystemExit):
        build_provider_llm("nope", PROVIDERS)


def test_mcq_rows_record_the_llm_base_url(tmp_path):
    s = ChunkStore(tmp_path / "eval.sqlite")
    s.add_chunks([Chunk("q", "d", "What is TF IDF? 1. a 2. b", "q", "mcq", 0, {"qid": "1", "answer": "1"}, "q:1", "W03")])
    s.close()
    rows = evaluate_mcq(tmp_path / "eval.sqlite", retriever_kind="-", llm=EchoLLM())
    assert rows[0].base_url == "offline" and rows[0].system == "no_rag"


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        from types import SimpleNamespace

        self.kwargs = kwargs
        msg = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _fake(llm):
    from types import SimpleNamespace

    fake = _FakeCompletions()
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return fake


def test_deepseek_thinking_switch_is_sent_as_extra_body():
    llm = LLMClient(model="deepseek-flash", base_url="https://api.deepseek.com", api_key="sk-test", thinking="disabled")
    fake = _fake(llm)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert fake.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_no_thinking_field_when_unset():
    llm = LLMClient(model="qwen-plus", base_url="https://example.invalid", api_key="sk-test", thinking="")
    fake = _fake(llm)
    llm.chat([{"role": "user", "content": "hi"}])
    assert fake.kwargs["extra_body"] is None


def test_provider_thinking_comes_from_config(monkeypatch):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "sk-test")
    providers = {"deepseek": {**PROVIDERS["deepseek"], "thinking": "disabled"}}
    assert build_provider_llm("deepseek", providers).thinking == "disabled"
