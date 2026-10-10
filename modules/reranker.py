from typing import List, Dict

from config import RERANK_TOP_K

# 懒加载的 CrossEncoder 模型实例
# 旧版本在模块导入时就加载模型（import 即下载、占内存），
# 现在改为首次调用 rerank_chunks 时才加载。
_reranker_model = None


def _get_reranker_model():
    """
    懒加载 BGE reranker 模型，并自动检测 CUDA。
    """
    global _reranker_model

    if _reranker_model is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker_model = CrossEncoder("BAAI/bge-reranker-base", device=device)

        print(f"当前 reranker 设备: {device}")

    return _reranker_model


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

    model = _get_reranker_model()

    # 构造 query-document 对
    # CrossEncoder 会对每一对 (query, chunk_text) 直接打分
    pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]

    # 计算每个 pair 的相关性得分
    scores = model.predict(pairs)

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
