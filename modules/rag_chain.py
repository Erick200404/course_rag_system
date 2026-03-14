from typing import List, Dict
from pathlib import Path

from config import VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_K
from config import CHAT_MODEL
from config import LLM_BACKEND, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
import os
import requests

from openai import OpenAI
from dotenv import load_dotenv

from modules.hybrid_retriever import hybrid_retrieve
from modules.reranker import rerank_chunks

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


def build_reference_text(retrieved_chunks: List[Dict]) -> str:
    """
    根据检索到的 chunks 生成参考来源信息。
    按文件名分组，并展示对应页码。
    """
    # 用于按 source 分组保存页码
    source_pages = {}

    for chunk in retrieved_chunks:
        source = chunk["source"]
        page = chunk["page"]

        if source not in source_pages:
            source_pages[source] = set()

        source_pages[source].add(page)

    reference_lines = []

    for source, pages in source_pages.items():
        # 去掉 .pdf 后缀，让显示更简洁
        source_name = Path(source).stem

        # 页码排序后拼接
        pages_sorted = sorted(pages)
        page_text = "、".join([f"第{p}页" for p in pages_sorted])

        reference_lines.append(f"{source_name}：{page_text}")

    return "\n".join(reference_lines)


def call_api_llm(prompt: str) -> str:
    """
    调用在线 OpenAI-compatible 聊天模型。
    """
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个课程资料问答助手，擅长根据给定资料回答问题。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def call_ollama_llm(prompt: str) -> str:
    """
    调用本地 Ollama 文本生成接口。
    """
    # 对当前单轮 RAG 场景，使用 /api/generate 更简单、更稳定
    url = f"{OLLAMA_BASE_URL}/api/generate"

    # 将 system 指令和用户 prompt 合并成一个完整提示词
    full_prompt = (
        "你是一个课程资料问答助手，擅长根据给定资料回答问题。\n\n"
        f"{prompt}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            # 保持和原在线模型一致的温度参数
            "temperature": 0.2,
            # 显式限制上下文窗口，降低本地 7B 模型的压力
            "num_ctx": 2048
        }
    }

    # 向本地 Ollama 发送请求
    response = requests.post(
        url,
        json=payload,
        timeout=OLLAMA_TIMEOUT
    )

    # 如果请求失败，这里尽量打印 Ollama 返回的详细错误信息
    if response.status_code != 200:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text

        raise RuntimeError(
            f"Ollama 调用失败，status_code={response.status_code}，detail={error_detail}"
        )

    # 解析 Ollama 返回的 JSON 结果
    data = response.json()

    # /api/generate 的回答正文在 response 字段中
    return data["response"]

def call_llm(prompt: str) -> str:
    """
    根据配置选择不同的 LLM backend。
    """
    # 如果配置为 ollama，则调用本地模型
    if LLM_BACKEND == "ollama":
        return call_ollama_llm(prompt)

    # 如果配置为 api，则调用在线模型
    if LLM_BACKEND == "api":
        return call_api_llm(prompt)

    # 其他非法配置，直接报错
    raise ValueError(f"不支持的 LLM_BACKEND: {LLM_BACKEND}")


def generate_answer(query: str, index, metadata: List[Dict], top_k: int = 3) -> Dict:
    """
    完整 RAG 流程：
    1. 使用 Hybrid Retrieval 召回候选 chunks
    2. 使用 Reranker 对候选 chunks 进行重排序
    3. 拼接上下文
    4. 调用大模型生成答案
    """

    # 第一步：先用 Hybrid Retrieval 召回更多候选片段
    # 这里先召回候选结果，给后续 Reranker 提供重排空间
    candidate_chunks = hybrid_retrieve(
        question=query,
        index=index,
        metadata=metadata,
        top_k_vector=VECTOR_TOP_K,
        top_k_bm25=BM25_TOP_K,
        final_top_k=FINAL_TOP_K
    )

    # 第二步：对候选结果进行重排序，选出最终最相关的 top_k 个 chunk
    retrieved_chunks = rerank_chunks(
        query=query,
        retrieved_chunks=candidate_chunks,
        top_k=top_k
    )

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
4. 在回答最后给出参考来源，按“文件名：第x页、第y页”的格式展示。
"""

    # 调用聊天模型生成最终答案
    answer = call_llm(prompt)

    # 生成参考来源文本
    reference_text = build_reference_text(retrieved_chunks)

    # 如果模型回答里没有主动给出参考来源，这里在末尾补上
    if "参考来源" not in answer:
        answer = answer.strip() + "\n\n参考来源：\n" + reference_text

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