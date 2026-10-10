from typing import Any

import modules.hybrid_retriever as hybrid_retriever
from modules.hybrid_retriever import HybridRetriever


class FakeChromaView:
    is_chroma_view = True

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": 0,
                "text": "检索增强生成用于课程资料问答",
                "page": 1,
                "source": "lecture.pdf",
                "score": 0.9,
            }
        ][:top_k]


def test_hybrid_retriever_supports_chroma_view(monkeypatch):
    monkeypatch.setattr(hybrid_retriever, "get_query_embedding", lambda _: [0.1, 0.2])
    metadata = [
        {
            "chunk_id": 0,
            "text": "检索增强生成用于课程资料问答",
            "page": 1,
            "source": "lecture.pdf",
        },
        {
            "chunk_id": 1,
            "text": "数据库索引可以加速查询",
            "page": 2,
            "source": "lecture.pdf",
        },
    ]

    retriever = HybridRetriever(FakeChromaView(), metadata)
    results = retriever.retrieve("什么是检索增强生成？")

    assert results
    assert results[0]["source"] == "lecture.pdf"
    assert results[0].get("score") == 0.9
