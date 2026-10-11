"""
API 编排路由单元测试：RAG_FLOW=crag / simple 的分派与多轮历史传递。

CRAG 与检索都通过 monkeypatch 伪造，不依赖真实模型与向量库。
"""

import pytest

import api.main as api_main


@pytest.fixture
def stub_pipeline(monkeypatch):
    """桩掉建库流程与两种执行路径，只验证分派逻辑。"""
    calls = {}

    # chat 在分派前会校验文件是否存在，这里只测路由逻辑，桩掉存在性检查
    monkeypatch.setattr(api_main.os.path, "exists", lambda p: True)

    monkeypatch.setattr(
        api_main,
        "build_rag_pipeline",
        lambda doc_paths: (None, [], "kb-id", "retriever"),
    )

    monkeypatch.setattr(
        api_main,
        "run_crag",
        lambda **kwargs: {
            "question": kwargs["question"],
            "original_question": kwargs["question"],
            "chunks": [{"chunk_id": 0, "text": "片段", "page": 1, "source": "a.pdf"}],
            "answer": "CRAG 答案",
            "retries": 0,
            "fallback": False,
            "_kwargs": kwargs,
        },
    )

    monkeypatch.setattr(
        api_main,
        "generate_answer",
        lambda **kwargs: {
            "query": kwargs["query"],
            "answer": "simple 答案",
            "retrieved_chunks": [],
        },
    )

    calls["history"] = None
    return calls


class TestFlowRouting:
    def test_crag_flow_returns_crag_fields(self, stub_pipeline, monkeypatch):
        monkeypatch.setattr(api_main, "RAG_FLOW", "crag")

        result = api_main.chat(
            api_main.ChatRequest(question="讲了什么？", doc_names=["a.pdf"])
        )

        assert result["flow"] == "crag"
        assert result["answer"] == "CRAG 答案"
        assert result["retries"] == 0
        assert result["fallback"] is False
        assert result["original_question"] == "讲了什么？"

    def test_simple_flow_returns_simple_fields(self, stub_pipeline, monkeypatch):
        monkeypatch.setattr(api_main, "RAG_FLOW", "simple")

        result = api_main.chat(
            api_main.ChatRequest(question="讲了什么？", doc_names=["a.pdf"])
        )

        assert result["flow"] == "simple"
        assert result["answer"] == "simple 答案"
        # simple 流程没有 CRAG 专有字段
        assert "retries" not in result

    def test_unknown_flow_raises(self, stub_pipeline, monkeypatch):
        from fastapi import HTTPException

        monkeypatch.setattr(api_main, "RAG_FLOW", "unknown")

        with pytest.raises(ValueError):
            api_main.chat(api_main.ChatRequest(question="x", doc_names=["a.pdf"]))


class TestHistoryPassing:
    def test_history_forwarded_to_crag(self, stub_pipeline, monkeypatch):
        monkeypatch.setattr(api_main, "RAG_FLOW", "crag")

        captured = {}

        def fake_run_crag(**kwargs):
            captured.update(kwargs)
            return {
                "question": "改写后",
                "original_question": kwargs["question"],
                "chunks": [],
                "answer": "答案",
                "retries": 1,
                "fallback": False,
            }

        monkeypatch.setattr(api_main, "run_crag", fake_run_crag)

        result = api_main.chat(
            api_main.ChatRequest(
                question="它指什么？",
                doc_names=["a.pdf"],
                history=[
                    {"question": "什么是 RAG？", "answer": "检索增强生成。"},
                ],
            )
        )

        # 历史事件已传给 CRAG 用于查询改写（可能是 pydantic 模型或 dict）
        assert captured["history"]
        first = captured["history"][0]
        first_question = first.question if hasattr(first, "question") else first["question"]
        assert first_question == "什么是 RAG？"
        # 改写后的问题在响应中体现
        assert result["question"] == "改写后"
        assert result["retries"] == 1
