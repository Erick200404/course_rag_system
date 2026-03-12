from typing import List, Dict
import numpy as np

from openai import OpenAI
from dotenv import load_dotenv
import os

# 加载 .env
load_dotenv()

# 初始化 OpenAI 兼容客户端
client = OpenAI(
    api_key=os.getenv("ZHI_API_KEY"),
    base_url=os.getenv("ZHI_BASE_URL"),
)


def embed_query(query: str) -> np.ndarray:
    """
    将用户问题转换为向量。
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    embedding = response.data[0].embedding

    # FAISS 通常要求 float32
    return np.array([embedding], dtype="float32")


def search_similar_chunks(query: str, index, metadata: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    根据用户问题，在 FAISS 中检索最相近的 top-k 个 chunk。

    参数:
        query: 用户问题
        index: FAISS 索引
        metadata: 与索引对应的元数据
        top_k: 返回前几个最相关结果

    返回:
        检索结果列表
    """

    # 1. 将问题转换为向量
    query_vector = embed_query(query)

    # 2. 在 FAISS 中搜索最相近的 top_k 个向量
    distances, indices = index.search(query_vector, top_k)

    results = []

    # distances 和 indices 都是二维数组，这里取第 0 行
    for score, idx in zip(distances[0], indices[0]):

        # FAISS 可能返回无效下标，保险起见做一下判断
        if idx < 0 or idx >= len(metadata):
            continue

        # 找到对应 metadata
        item = metadata[idx].copy()

        # 把距离分数也带上，方便后面调试
        item["score"] = float(score)

        results.append(item)

    return results


if __name__ == "__main__":
    from modules.pdf_loader import load_pdf
    from modules.text_splitter import split_text
    from modules.embeddings import embed_chunks
    from modules.vector_store import build_faiss_index

    # 1. 读取 PDF
    pages = load_pdf("data/test.pdf")

    # 2. 切分 chunk
    chunks = split_text(pages)

    # 3. 生成 embedding
    embedded_chunks = embed_chunks(chunks)

    # 4. 建立 FAISS 索引
    index, metadata = build_faiss_index(embedded_chunks)

    # 5. 测试问题
    query = "BETL 的总体框架是什么？"

    results = search_similar_chunks(query, index, metadata, top_k=3)

    print(f"用户问题: {query}\n")
    print("检索结果：\n")

    for i, item in enumerate(results, start=1):
        print(f"第{i}条")
        print(f"chunk_id: {item['chunk_id']}")
        print(f"page: {item['page']}")
        print(f"source: {item['source']}")
        print(f"score: {item['score']}")
        print(f"text preview: {item['text'][:200]}")
        print("-" * 50)