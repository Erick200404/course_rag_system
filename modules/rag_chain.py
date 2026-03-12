from typing import List, Dict
import os

from openai import OpenAI
from dotenv import load_dotenv

from modules.retriever import search_similar_chunks

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 兼容客户端
client = OpenAI(
    api_key=os.getenv("ZHI_API_KEY"),
    base_url=os.getenv("ZHI_BASE_URL"),
)


def build_context(retrieved_chunks: List[Dict]) -> str:
    """
    将检索到的 chunks 拼接成可供大模型阅读的上下文。
    """

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        part = (
            f"资料{i}（来源: {chunk['source']}，第{chunk['page']}页）:\n"
            f"{chunk['text']}"
        )
        context_parts.append(part)

    return "\n\n".join(context_parts)


def generate_answer(query: str, index, metadata: List[Dict], top_k: int = 3) -> Dict:
    """
    完整 RAG 流程：
    1. 根据问题检索相关 chunks
    2. 拼接上下文
    3. 调用大模型生成答案
    """

    # 先召回最相关的 chunk
    retrieved_chunks = search_similar_chunks(query, index, metadata, top_k=top_k)

    # 将召回内容拼成 context
    context = build_context(retrieved_chunks)

    # 构造 prompt
    prompt = f"""
请根据以下课程资料回答问题。

{context}

问题：
{query}

回答要求：
1. 只能根据提供的资料回答，不要随意补充资料中没有的信息。
2. 如果资料不足以回答问题，请明确说“根据当前检索到的资料，无法完整回答该问题”。
3. 回答要清晰、准确、简洁。
4. 在回答最后给出参考页码，格式示例：参考页码：第2页、第3页。
"""

    # 调用聊天模型
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个课程资料问答助手，擅长根据给定资料回答问题。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks
    }


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

    # 4. 构建 FAISS 索引
    index, metadata = build_faiss_index(embedded_chunks)

    # 5. 提问
    query = "BETL 的总体框架是什么？"

    result = generate_answer(query, index, metadata, top_k=3)

    print(f"用户问题：{result['query']}\n")
    print("模型回答：")
    print(result["answer"])
    print("\n召回片段预览：")
    print("-" * 50)

    for i, item in enumerate(result["retrieved_chunks"], start=1):
        print(f"第{i}条：来源 {item['source']} 第{item['page']}页")
        print(item["text"][:150])
        print("-" * 50)