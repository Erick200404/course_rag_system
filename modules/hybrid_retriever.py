from typing import Any, Dict, List

from modules.vector_store import search_faiss_index
from modules.embeddings import get_query_embedding
from modules.bm25_retriever import BM25Retriever

from config import VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_K


class HybridRetriever:
    """
    混合检索器：向量检索 + BM25 关键词检索。

    修复说明：
    旧版本使用模块级全局单例缓存 BM25Retriever，导致切换到另一组 PDF
    构建的新知识库后，BM25 仍然基于旧 metadata 检索，返回错误结果。

    现在 BM25 索引的生命周期与本实例绑定：
    - 构造时基于传入的 metadata 构建一次 BM25
    - 每个知识库（kb_id）持有独立的 HybridRetriever 实例
    - 销毁实例即释放对应 BM25 索引
    """

    def __init__(self, index: Any, metadata: List[Dict[str, Any]]):
        """
        参数:
            index: FAISS 索引或 ChromaVectorView
            metadata: 与向量一一对应的元数据列表
        """
        self.index = index
        self.metadata = metadata

        # BM25 索引与 metadata 绑定，随实例创建、随实例销毁
        self.bm25 = BM25Retriever(metadata)

    def retrieve(
        self,
        question: str,
        top_k_vector: int = VECTOR_TOP_K,
        top_k_bm25: int = BM25_TOP_K,
        final_top_k: int = FINAL_TOP_K,
    ) -> List[Dict]:
        """
        混合检索：结合向量检索和 BM25 关键词检索。

        参数:
            question: 用户问题
            top_k_vector: 向量检索召回数量
            top_k_bm25: BM25 检索召回数量
            final_top_k: 最终返回数量

        返回:
            合并去重后的检索结果列表
        """
        # 1. 向量检索
        query_embedding = get_query_embedding(question)

        if getattr(self.index, "is_chroma_view", False):
            vector_results = self.index.search(
                query_embedding,
                top_k=top_k_vector,
            )
        else:
            vector_results = search_faiss_index(
                index=self.index,
                metadata=self.metadata,
                query_embedding=query_embedding,
                top_k=top_k_vector,
            )

        # 2. BM25 检索
        bm25_results = self.bm25.search(
            question,
            top_k=top_k_bm25,
        )

        # 3. 合并结果
        # 用 (source, page, chunk_id) 作为唯一标识去重
        merged = {}

        for item in vector_results:
            key = (item["source"], item["page"], item["chunk_id"])
            merged[key] = item

        for item in bm25_results:
            key = (item["source"], item["page"], item["chunk_id"])
            if key not in merged:
                merged[key] = item

        # 4. 转成列表并截断
        # 当前阶段先不做统一排序，由 reranker 负责重排
        results = list(merged.values())

        return results[:final_top_k]
