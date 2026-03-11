from typing import List, Dict
import os

from openai import OpenAI
from dotenv import load_dotenv

# 读取 .env 文件
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("ZHI_API_KEY"),
    base_url=os.getenv("ZHI_BASE_URL"),
)


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    调用 embedding API，将 chunk 文本转为向量。

    参数:
        chunks: text_splitter 生成的 chunk 列表

    返回:
        带 embedding 向量的 chunk 列表
    """

    embedded_chunks = []

    for chunk in chunks:

        text = chunk["text"]

        # 调用 embedding 接口
        response = client.embeddings.create(
            model="text-embedding-3-small",  # 常见 embedding 模型
            input=text
        )

        # 获取向量
        embedding = response.data[0].embedding

        # 复制原 chunk
        new_chunk = chunk.copy()

        # 加入向量
        new_chunk["embedding"] = embedding

        embedded_chunks.append(new_chunk)

    return embedded_chunks


if __name__ == "__main__":

    from modules.pdf_loader import load_pdf
    from modules.text_splitter import split_text

    # 1. 读取 PDF
    pages = load_pdf("data/test.pdf")

    # 2. 切分 chunk
    chunks = split_text(pages)

    # 3. 调用 embedding
    embedded_chunks = embed_chunks(chunks)

    print(f"生成 {len(embedded_chunks)} 个 embedding\n")

    # 打印前两个
    for item in embedded_chunks[:2]:
        print(f"chunk_id: {item['chunk_id']}")
        print(f"page: {item['page']}")
        print(f"text preview: {item['text'][:100]}")
        print(f"embedding dim: {len(item['embedding'])}")
        print("-" * 50)