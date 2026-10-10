# api/main.py

import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import (
    RERANK_TOP_K,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    HF_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INDEX_SCHEMA_VERSION,
    PDF_PARSE_BACKEND,
)

from modules.doc_registry import compute_kb_id, register_documents
from modules.pdf_loader import load_pdf
from modules.text_splitter import split_text
from modules.data_cleaner import filter_pages
from modules.embeddings import embed_chunks
from modules.hybrid_retriever import HybridRetriever
from modules.vector_store import (
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    faiss_index_exists,
)
from modules.rag_chain import generate_answer


# 创建 FastAPI 应用
app = FastAPI(
    title="Course RAG API",
    description="课程 PDF 智能问答系统 API",
    version="1.0.0"
)


# ==============================
# HybridRetriever 实例缓存
# ==============================
# kb_id -> HybridRetriever
# 修复说明：旧版 BM25 使用模块级全局单例，切换知识库后不重建；
# 现在每个知识库持有独立的 retriever 实例，BM25 随之正确切换。
_retriever_cache: Dict[str, HybridRetriever] = {}


# ==============================
# 请求数据结构
# ==============================
class ChatRequest(BaseModel):
    # 用户问题
    question: str = Field(..., description="用户输入的问题")

    # 要使用的 PDF 文件名列表（文件需已存在于 data 目录中）
    pdf_names: List[str] = Field(..., description="参与问答的 PDF 文件名列表")

    # 最终送入大模型的片段数量
    top_k: int = Field(default=RERANK_TOP_K, description="最终召回片段数量")


# ==============================
# 健康检查接口
# ==============================
@app.get("/health")
def health_check():
    """
    用于检查 API 服务是否正常运行。
    """
    return {"status": "ok"}


def get_current_embedding_model() -> str:
    """
    根据当前 embedding backend 返回实际使用的模型名。
    """
    if EMBEDDING_BACKEND == "hf":
        return HF_EMBEDDING_MODEL
    return EMBEDDING_MODEL


def get_kb_storage_paths(kb_id: str) -> tuple[str, str]:
    """
    根据知识库 id 生成索引存储路径。

    kb_id 由 PDF 文件内容 sha256 + embedding 配置共同决定：
    - 同一组文件（内容不变）+ 同一 embedding 模型 -> 复用本地索引
    - 任何文件内容更新或切换 embedding 模型 -> 新 kb_id -> 自动重建
    """
    storage_dir = os.path.join("storage", kb_id)

    index_path = os.path.join(storage_dir, "faiss_index.bin")
    metadata_path = os.path.join(storage_dir, "metadata.pkl")

    return index_path, metadata_path


def build_rag_pipeline(pdf_paths: List[str]):
    """
    根据多个 PDF 构建或加载完整的 RAG 检索基础。

    逻辑：
    1. 将文件登记进 SQLite 注册表（记录 sha256/大小/时间）
    2. 基于文件内容 hash + embedding 配置计算 kb_id
    3. 如果本地已有该 kb_id 的联合索引，则直接加载
    4. 否则逐个读取 PDF、过滤噪声页、切分 chunk
    5. 合并所有 PDF 的 chunk，统一生成 embedding、构建 FAISS 索引并保存
    6. 为该知识库构建（或复用）HybridRetriever 实例

    返回:
        (index, metadata, kb_id, retriever)
    """
    # 1. 登记文档信息
    register_documents(pdf_paths)

    # 2. 计算内容感知的知识库 id
    kb_id = compute_kb_id(
        pdf_paths,
        embedding_backend=EMBEDDING_BACKEND,
        embedding_model=get_current_embedding_model(),
        index_schema_version=(
            f"{INDEX_SCHEMA_VERSION}:{PDF_PARSE_BACKEND}:"
            f"{CHUNK_SIZE}:{CHUNK_OVERLAP}"
        ),
    )

    index_path, metadata_path = get_kb_storage_paths(kb_id)

    # 打印当前知识库 id，便于调试
    print(f"当前知识库 kb_id: {kb_id}")

    # 3. 如果索引已存在，则直接加载
    if faiss_index_exists(index_path, metadata_path):
        index, metadata = load_faiss_index(index_path, metadata_path)

    # 4. 如果索引不存在，则从头构建
    else:
        all_chunks = []

        # 按文件名排序，保证同一组 PDF 生成确定性的全局 chunk_id
        next_chunk_id = 0

        for pdf_path in sorted(pdf_paths, key=lambda p: os.path.basename(p).lower()):
            # 读取 PDF 内容（markdown / text 后端由配置决定）
            pages = load_pdf(pdf_path)

            # 过滤噪声页
            pages = filter_pages(pages)

            # 切分 chunk
            chunks = split_text(
                pages,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                backend=PDF_PARSE_BACKEND,
                start_chunk_id=next_chunk_id,
            )

            # 合并到总 chunk 列表
            all_chunks.extend(chunks)
            next_chunk_id += len(chunks)

        # 为所有 chunk 统一生成 embedding
        embedded_chunks = embed_chunks(all_chunks)

        # 构建 FAISS 索引
        index, metadata = build_faiss_index(embedded_chunks)

        # 保存索引和 metadata，方便下次复用
        save_faiss_index(index, metadata, index_path, metadata_path)

    # 5. 构建或复用当前知识库的 HybridRetriever
    retriever = _retriever_cache.get(kb_id)
    if retriever is None:
        retriever = HybridRetriever(index, metadata)
        _retriever_cache[kb_id] = retriever

    return index, metadata, kb_id, retriever


# ==============================
# 问答接口
# ==============================
@app.post("/chat")
def chat(request: ChatRequest):
    """
    根据指定 PDF 文件和问题，返回 RAG 问答结果。
    """
    # 去掉问题首尾空格
    question = request.question.strip()

    # 检查问题是否为空
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    # 检查 PDF 文件名列表是否为空
    if not request.pdf_names:
        raise HTTPException(status_code=400, detail="pdf_names 不能为空")

    # 将文件名转为 data 目录下的路径
    pdf_paths = []
    for pdf_name in request.pdf_names:
        if Path(pdf_name).name != pdf_name:
            raise HTTPException(
                status_code=400,
                detail="pdf_names 中只能包含文件名，不能包含路径"
            )

        pdf_path = os.path.join("data", pdf_name)

        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在: {pdf_name}"
            )

        pdf_paths.append(pdf_path)

    # 构建或加载知识库索引
    index, metadata, kb_id, retriever = build_rag_pipeline(pdf_paths)

    # 调用 RAG 逻辑生成答案
    result = generate_answer(
        query=question,
        retriever=retriever,
        top_k=request.top_k
    )

    # 返回结果
    return {
        "question": result["query"],
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"]
    }
