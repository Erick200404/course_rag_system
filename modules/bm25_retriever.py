from typing import List, Dict
from rank_bm25 import BM25Okapi
from config import BM25_TOP_K
import jieba   # 新增：中文分词


class BM25Retriever:
    """
    基于 BM25 的关键词检索器。

    BM25 是一种经典的信息检索算法，
    常用于搜索引擎（如 Elasticsearch）。
    """

    def __init__(self, metadata: List[Dict]):
        """
        初始化 BM25 检索器。

        参数：
            metadata: 向量库对应的 metadata 列表
        """

        self.metadata = metadata

        # 从 metadata 中提取文本
        corpus = [item["text"] for item in metadata]

        # ==========================
        # 修改：使用 jieba 中文分词
        # ==========================
        # 如果直接使用 split()，中文基本不会被正确切分
        # jieba 可以把句子切成更合理的词
        tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]

        # 构建 BM25 索引
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> List[Dict]:
        """
        根据 query 执行 BM25 检索。

        参数：
            query: 用户问题
            top_k: 返回最相关的 chunk 数量
        """

        # ==========================
        # 修改：query 也使用 jieba 分词
        # ==========================
        tokenized_query = list(jieba.cut(query))

        # 获取 BM25 得分
        scores = self.bm25.get_scores(tokenized_query)

        # 按得分排序
        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for idx, score in ranked[:top_k]:

            item = self.metadata[idx].copy()

            # 添加 BM25 得分（方便调试）
            item["bm25_score"] = float(score)

            results.append(item)

        return results