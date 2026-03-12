from typing import List, Dict
from config import RERANK_TOP_K
from sentence_transformers import CrossEncoder


# 这里使用一个常见的中文/英文都比较常用的 reranker 模型
# 第一次加载会稍慢，因为需要下载模型
reranker_model = CrossEncoder("BAAI/bge-reranker-base")


def rerank_chunks(query: str, retrieved_chunks: List[Dict], top_k: int = RERANK_TOP_K) -> List[Dict]:
    """
    使用 Cross-Encoder 对召回到的 chunk 进行重排序。

    参数:
        query: 用户问题
        retrieved_chunks: Hybrid Retrieval 召回到的候选 chunk 列表
        top_k: 最终保留的 chunk 数量

    返回:
        rerank 后的 top_k 个 chunk
    """
    # 如果没有召回结果，直接返回空列表
    if not retrieved_chunks:
        return []

    # 构造 query-document 对
    # CrossEncoder 会对每一对 (query, chunk_text) 直接打分
    pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]

    # 计算每个 pair 的相关性得分
    scores = reranker_model.predict(pairs)

    # 将得分写回每个 chunk
    reranked_results = []
    for chunk, score in zip(retrieved_chunks, scores):
        new_chunk = chunk.copy()
        new_chunk["rerank_score"] = float(score)
        reranked_results.append(new_chunk)

    # 按 rerank_score 从高到低排序
    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

    # 返回最终 top_k 个结果
    return reranked_results[:top_k]