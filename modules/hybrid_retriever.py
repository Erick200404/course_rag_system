from typing import List, Dict

from modules.vector_store import search_faiss_index
from modules.embeddings import get_query_embedding
from modules.bm25_retriever import BM25Retriever


def hybrid_retrieve(
    question: str,
    index,
    metadata: List[Dict],
    top_k_vector: int = 5,
    top_k_bm25: int = 5,
    final_top_k: int = 5,
) -> List[Dict]:
    """
    混合检索：结合向量检索和 BM25 关键词检索。

    参数:
        question: 用户问题
        index: FAISS 索引
        metadata: 与向量对应的元数据列表
        top_k_vector: 向量检索召回数量
        top_k_bm25: BM25 检索召回数量
        final_top_k: 最终返回数量

    返回:
        合并去重后的检索结果列表
    """

    # 1. 先做向量检索
    query_embedding = get_query_embedding(question)
    vector_results = search_faiss_index(
        index=index,
        metadata=metadata,
        query_embedding=query_embedding,
        top_k=top_k_vector
    )

    # 2. 再做 BM25 检索
    bm25_retriever = BM25Retriever(metadata)
    bm25_results = bm25_retriever.search(question, top_k=top_k_bm25)

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

    # 4. 转成列表
    results = list(merged.values())

    # 5. 先简单截断
    # 当前阶段先不做统一排序，后面 reranker 会专门负责重排
    return results[:final_top_k]