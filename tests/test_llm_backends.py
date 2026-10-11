"""
llm_backends 单元测试：LCEL 链的组装与调用。

使用 LangChain 自带的 FakeListChatModel，不依赖真实的 OpenAI / Ollama 服务。
"""

import pytest

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import modules.llm_backends as llm_backends
import modules.rag_chain as rag_chain
from modules.llm_backends import (
    SYSTEM_PROMPT,
    build_answer_chain,
    build_chat_model,
    invoke_llm,
)


class RecordingChatModel(FakeListChatModel):
    """记录最后一次收到的消息，用于断言提示词组装结果。"""

    received: list = []

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        RecordingChatModel.received = messages
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture(autouse=True)
def reset_recorder():
    RecordingChatModel.received = []
    yield
    RecordingChatModel.received = []


def test_chain_returns_string_output():
    chain = build_answer_chain(FakeListChatModel(responses=["这是模型生成的答案"]))

    result = chain.invoke({"prompt": "问题：什么是检索增强生成？"})

    assert result == "这是模型生成的答案"


def test_prompt_template_includes_system_and_human_messages():
    model = RecordingChatModel(responses=["ok"])

    invoke_llm("资料1：检索增强生成", chat_model=model)

    messages = RecordingChatModel.received
    assert len(messages) == 2

    roles = [message.type for message in messages]
    assert roles == ["system", "human"]

    assert messages[0].content == SYSTEM_PROMPT
    assert "检索增强生成" in messages[1].content


def test_invoke_llm_returns_text():
    result = invoke_llm("随便一个问题", chat_model=FakeListChatModel(responses=["答案文本"]))

    assert result == "答案文本"


def test_unsupported_backend_raises(monkeypatch):
    monkeypatch.setattr(llm_backends, "LLM_BACKEND", "unknown-backend")

    with pytest.raises(ValueError):
        build_chat_model()


def test_api_backend_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm_backends, "LLM_BACKEND", "api")
    monkeypatch.delenv("ZHI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        build_chat_model()


def test_api_backend_builds_chat_openai(monkeypatch):
    monkeypatch.setattr(llm_backends, "LLM_BACKEND", "api")
    monkeypatch.setenv("ZHI_API_KEY", "test-key")
    monkeypatch.setenv("ZHI_BASE_URL", "https://example.com/v1")

    model = build_chat_model()

    assert model.model_name == llm_backends.CHAT_MODEL
    assert model.temperature == 0.2


def test_ollama_backend_builds_chat_ollama(monkeypatch):
    monkeypatch.setattr(llm_backends, "LLM_BACKEND", "ollama")

    model = build_chat_model()

    assert model.model == llm_backends.OLLAMA_MODEL
    assert model.num_ctx == 2048
    # 超时通过 httpx client kwargs 传递
    assert model.client_kwargs["timeout"] == llm_backends.OLLAMA_TIMEOUT


def test_generate_answer_uses_lcel_chain(monkeypatch):
    """
    端到端验证 generate_answer 走 LCEL 链：
    伪造检索器与模型，确认答案来自链的输出。
    """

    class FakeRetriever:
        def retrieve(self, question, top_k_vector, top_k_bm25, final_top_k):
            return [
                {
                    "chunk_id": 0,
                    "text": "检索增强生成用于课程资料问答",
                    "page": 3,
                    "source": "lecture.pdf",
                }
            ]

    def fake_rerank(query, retrieved_chunks, top_k):
        return retrieved_chunks[:top_k]

    monkeypatch.setattr(rag_chain, "rerank_chunks", fake_rerank)
    monkeypatch.setattr(
        rag_chain,
        "invoke_llm",
        lambda prompt: "这是基于资料生成的答案\n\n参考来源：\nlecture：第3页",
    )

    result = rag_chain.generate_answer("讲了什么？", FakeRetriever(), top_k=1)

    assert result["answer"].startswith("这是基于资料生成的答案")
    assert result["retrieved_chunks"][0]["source"] == "lecture.pdf"
