from typing import List, Dict
from pathlib import Path

from config import VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_K

from modules.reranker import rerank_chunks
from modules.llm_backends import invoke_llm


def build_context(retrieved_chunks: List[Dict]) -> str:
    """
    将检索到的 chunks 拼接成可供大模型阅读的上下文。
    markdown 后端产生的 chunk 带章节路径，一并展示，帮助模型定位。
    """
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        location = f"来源: {chunk['source']}，第{chunk['page']}页"

        # 如果有章节路径，附加展示
        if chunk.get("heading"):
            location += f"，章节: {chunk['heading']}"

        part = (
            f"资料{i}（{location}）:\n"
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


def generate_answer(query: str, retriever, top_k: int = 3) -> Dict:
    """
    完整 RAG 流程：
    1. 使用 HybridRetriever 召回候选 chunks
    2. 使用 Reranker 对候选 chunks 进行重排序
    3. 拼接上下文
    4. 调用大模型生成答案

    参数:
        query: 用户问题
        retriever: HybridRetriever 实例（与当前知识库绑定）
        top_k: 最终保留的片段数量
    """

    # 第一步：先用 Hybrid Retrieval 召回更多候选片段
    # 这里先召回候选结果，给后续 Reranker 提供重排空间
    candidate_chunks = retriever.retrieve(
        question=query,
        top_k_vector=VECTOR_TOP_K,
        top_k_bm25=BM25_TOP_K,
        final_top_k=FINAL_TOP_K,
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

    # 通过 LCEL 链调用聊天模型生成最终答案
    answer = invoke_llm(prompt)

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
    from modules.hybrid_retriever import HybridRetriever

    # 1. 读取 PDF
    pages = load_pdf("data/test.pdf")

    # 2. 切分 chunk
    chunks = split_text(pages)

    # 3. 生成 embedding
    embedded_chunks = embed_chunks(chunks)

    # 4. 构建 FAISS 索引
    index, metadata = build_faiss_index(embedded_chunks)

    # 5. 构建混合检索器（与当前知识库绑定）
    retriever = HybridRetriever(index, metadata)

    # 6. 提问
    query = "BETL 的总体框架是什么？"

    result = generate_answer(query, retriever, top_k=3)

    print(f"用户问题：{result['query']}\n")
    print("模型回答：")
    print(result["answer"])
    print("\n召回片段预览：")
    print("-" * 50)

    for i, item in enumerate(result["retrieved_chunks"], start=1):
        print(f"第{i}条：来源 {item['source']} 第{item['page']}页")
        print(item["text"][:150])
        print("-" * 50)
