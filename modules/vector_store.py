from typing import List, Dict, Tuple
import numpy as np
import faiss


def build_faiss_index(embedded_chunks: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
    """
    根据带 embedding 的 chunk 列表构建 FAISS 索引。

    参数:
        embedded_chunks: 每个元素都包含 text/page/source/chunk_id/embedding

    返回:
        index: FAISS 索引对象
        metadata: 与向量一一对应的元数据列表
    """

    # 如果没有数据，直接报错
    if not embedded_chunks:
        raise ValueError("embedded_chunks 不能为空")

    # 取出所有 embedding，拼成二维 numpy 数组
    # FAISS 要求数据类型通常为 float32
    embeddings = np.array(
        [chunk["embedding"] for chunk in embedded_chunks],
        dtype="float32"
    )

    # 获取向量维度，比如 1536
    dimension = embeddings.shape[1]

    # 创建最基础的 FAISS 索引
    # IndexFlatL2 表示用 L2 距离（欧式距离）做相似度搜索
    index = faiss.IndexFlatL2(dimension)

    # 将所有向量加入索引
    index.add(embeddings)

    # 单独保存 metadata
    # 因为 FAISS 只存向量，不存原始文本和页码信息
    metadata = []
    for chunk in embedded_chunks:
        metadata.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"],
            }
        )

    return index, metadata


if __name__ == "__main__":
    from modules.pdf_loader import load_pdf
    from modules.text_splitter import split_text
    from modules.embeddings import embed_chunks

    # 1. 读取 PDF
    pages = load_pdf("data/test.pdf")

    # 2. 切分成 chunk
    chunks = split_text(pages)

    # 3. 调用 embedding API 生成向量
    embedded_chunks = embed_chunks(chunks)

    # 4. 构建 FAISS 索引
    index, metadata = build_faiss_index(embedded_chunks)

    # 打印一些调试信息，确认索引构建成功
    print("FAISS 索引构建成功")
    print(f"向量数量: {index.ntotal}")
    print(f"metadata 数量: {len(metadata)}")
    print("第一条 metadata 预览:")
    print(metadata[0])