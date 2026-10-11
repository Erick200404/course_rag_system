"""
rag_graph 单元测试：CRAG 分支、循环、兜底与多轮改写。

用 FakeListChatModel 控制评分/生成输出，验证图的路径选择，不依赖真实模型。
"""

import pytest

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import modules.rag_graph as rag_graph
from modules.rag_graph import (
    _format_history,
    build_rag_graph,
    grade_node,
    rewrite_node,
    route_after_grade,
    run_crag,
)


class FakeRetriever:
    """返回固定候选片段的检索器。"""

    DEFAULT_CHUNKS = [
        {
            "chunk_id": 0,
            "text": "检索增强生成用于课程资料问答",
            "page": 3,
            "source": "lecture.pdf",
        }
    ]

    def __init__(self, chunks=None):
        # 注意：不能写 chunks or DEFAULT_CHUNKS，空列表会被兜底成默认内容
        self.chunks = list(self.DEFAULT_CHUNKS) if chunks is None else chunks
        self.calls = []

    def retrieve(self, question, **kwargs):
        self.calls.append(question)
        return self.chunks


@pytest.fixture(autouse=True)
def stub_rerank(monkeypatch):
    """跳过 reranker 依赖，直接返回候选。"""
    monkeypatch.setattr(
        rag_graph, "rerank_chunks", lambda query, retrieved_chunks, top_k: retrieved_chunks[:top_k]
    )


@pytest.fixture(autouse=True)
def stub_invoke_llm(monkeypatch):
    """跳过生成链的真实调用。"""
    import modules.rag_chain as rag_chain

    monkeypatch.setattr(rag_chain, "invoke_llm", lambda prompt: "生成的答案")


class TestGraphStructure:
    def test_graph_compiles(self):
        graph = build_rag_graph()
        assert graph is not None

    def test_route_decisions(self):
        # 相关 → 直接生成
        assert route_after_grade({"relevant": True, "retries": 0}) == "generate"
        # 不相关且未耗尽重试 → 改写
        assert route_after_grade({"relevant": False, "retries": 0}) == "rewrite"
        # 不相关且重试耗尽 → 兜底
        assert (
            route_after_grade({"relevant": False, "retries": rag_graph.MAX_RETRIEVE_RETRIES})
            == "fallback"
        )


class TestGradeNode:
    def test_relevant_verdict(self, monkeypatch):
        monkeypatch.setattr(
            rag_graph, "build_chat_model", lambda: FakeListChatModel(responses=["relevant"])
        )

        result = grade_node(
            {"question": "讲了什么？", "chunks": [{"chunk_id": 0, "text": "正文", "page": 1, "source": "a.pdf"}]}
        )

        assert result["relevant"] is True

    def test_irrelevant_verdict(self, monkeypatch):
        monkeypatch.setattr(
            rag_graph, "build_chat_model", lambda: FakeListChatModel(responses=["irrelevant"])
        )

        result = grade_node(
            {"question": "讲了什么？", "chunks": [{"chunk_id": 0, "text": "正文", "page": 1, "source": "a.pdf"}]}
        )

        assert result["relevant"] is False

    def test_empty_chunks_is_irrelevant(self, monkeypatch):
        # 无召回时不应调用模型
        def must_not_call():
            raise AssertionError("无召回不应调用评分模型")

        monkeypatch.setattr(rag_graph, "build_chat_model", must_not_call)

        result = grade_node({"question": "讲了什么？", "chunks": []})

        assert result["relevant"] is False


class TestRewriteNode:
    def test_rewrites_question_and_counts_retry(self, monkeypatch):
        monkeypatch.setattr(
            rag_graph,
            "build_chat_model",
            lambda: FakeListChatModel(responses=["检索增强生成的整体流程是什么"]),
        )

        result = rewrite_node(
            {
                "question": "它讲了什么？",
                "history": [{"question": "什么是检索增强生成？", "answer": "一种结合检索的生成方法。"}],
                "retries": 0,
            }
        )

        assert result["question"] == "检索增强生成的整体流程是什么"
        assert result["retries"] == 1

    def test_empty_rewrite_keeps_original(self, monkeypatch):
        monkeypatch.setattr(
            rag_graph, "build_chat_model", lambda: FakeListChatModel(responses=[""])
        )

        result = rewrite_node({"question": "原始问题", "history": [], "retries": 1})

        assert result["question"] == "原始问题"
        assert result["retries"] == 2

    def test_history_formatting_keeps_recent_turns(self):
        history = [
            {"question": f"第{i}轮问题", "answer": f"第{i}轮回答"} for i in range(1, 8)
        ]

        formatted = _format_history(history)

        assert "第1轮问题" not in formatted
        assert "第7轮问题" in formatted

    def test_empty_history_placeholder(self):
        assert _format_history([]) == "（无历史）"


class TestCragFlow:
    def test_relevant_path_generates_without_retry(self, monkeypatch):
        """相关资料：一次检索直接生成，不改写。"""
        # 共享同一个假模型实例，responses 才会按顺序被消费
        model = FakeListChatModel(responses=["relevant"])
        monkeypatch.setattr(rag_graph, "build_chat_model", lambda: model)

        retriever = FakeRetriever()
        result = run_crag("讲了什么？", retriever, top_k=1)

        assert result["retries"] == 0
        assert result["fallback"] is False
        assert result["answer"] == "生成的答案"
        # 只检索了一次
        assert len(retriever.calls) == 1

    def test_irrelevant_then_relevant_retries_once(self, monkeypatch):
        """
        首次评分不相关 → 改写 → 重新检索 → 第二次评分相关 → 生成。
        模型依次输出：irrelevant、改写后问题、relevant
        """
        model = FakeListChatModel(responses=["irrelevant", "改写后的问题", "relevant"])
        monkeypatch.setattr(rag_graph, "build_chat_model", lambda: model)

        retriever = FakeRetriever()
        result = run_crag("它讲了什么？", retriever, top_k=1)

        assert result["retries"] == 1
        assert result["fallback"] is False
        assert result["question"] == "改写后的问题"
        # 检索了两次（原问题 + 改写后）
        assert len(retriever.calls) == 2

    def test_always_irrelevant_falls_back(self, monkeypatch):
        """
        反复评分不相关 → 改写重试耗尽 → 兜底回答。
        模型交替输出：irrelevant / 改写问题 / irrelevant / 改写问题 / irrelevant
        """
        model = FakeListChatModel(
            responses=["irrelevant", "改写1", "irrelevant", "改写2", "irrelevant"]
        )
        monkeypatch.setattr(rag_graph, "build_chat_model", lambda: model)

        retriever = FakeRetriever()
        result = run_crag("无关问题", retriever, top_k=1)

        assert result["fallback"] is True
        assert result["retries"] == rag_graph.MAX_RETRIEVE_RETRIES
        assert "无法完整回答" in result["answer"]
        # 初始 + 每次改写后各一次
        assert len(retriever.calls) == rag_graph.MAX_RETRIEVE_RETRIES + 1

    def test_empty_retrieval_falls_back_without_rewrite(self, monkeypatch):
        """
        检索不到任何内容 → 不调评分模型，也不做无意义的改写重试 → 直接兜底。
        """
        calls = {"n": 0}

        def must_not_call():
            calls["n"] += 1
            return FakeListChatModel(responses=["relevant"])

        monkeypatch.setattr(rag_graph, "build_chat_model", must_not_call)

        retriever = FakeRetriever(chunks=[])
        result = run_crag("完全无关的问题", retriever, top_k=1)

        assert result["fallback"] is True
        assert result["chunks"] == []
        # 改写也无意义，因此模型一次都不该调用
        assert calls["n"] == 0
        assert len(retriever.calls) == 1

    def test_empty_flag_skips_rewrite(self):
        """评分标记 empty 时不应走 rewrite 分支。"""
        assert route_after_grade({"relevant": False, "empty": True, "retries": 0}) == "fallback"
