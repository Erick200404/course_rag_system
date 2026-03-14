# api/main.py

import os
import re
import hashlib
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import RERANK_TOP_K

from modules.pdf_loader import load_pdf
from modules.text_splitter import split_text
from modules.data_cleaner import filter_pages
from modules.embeddings import embed_chunks
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


def get_multi_pdf_storage_paths(pdf_paths: List[str]) -> tuple[str, str]:
    """
    根据多个 PDF 文件路径生成联合索引存储路径。

    逻辑：
    1. 取所有 PDF 文件名（不带后缀）
    2. 排序后拼接，避免上传顺序不同导致重复建库
    3. 使用 md5 生成固定长度知识库 id
    4. 用该 id 作为 storage 子目录名称
    """
    # 取所有文件名（不带后缀）
    pdf_names = [Path(path).stem for path in pdf_paths]

    # 排序，避免相同文件集合因顺序不同生成不同索引
    pdf_names = sorted(pdf_names)

    # 拼接文件名
    joined_name = "_".join(pdf_names)

    # 过滤特殊字符，避免路径显示混乱
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", joined_name)

    # 使用 md5 生成稳定的知识库 id
    kb_id = hashlib.md5(safe_name.encode("utf-8")).hexdigest()

    # 存储目录
    storage_dir = os.path.join("storage", kb_id)

    # 索引文件路径
    index_path = os.path.join(storage_dir, "faiss_index.bin")

    # metadata 文件路径
    metadata_path = os.path.join(storage_dir, "metadata.pkl")

    return index_path, metadata_path


def build_rag_pipeline(pdf_paths: List[str]):
    """
    根据多个 PDF 构建或加载完整的 RAG 检索基础。

    逻辑：
    1. 如果本地已有联合索引，则直接加载
    2. 否则逐个读取 PDF、过滤噪声页、切分 chunk
    3. 合并所有 PDF 的 chunk
    4. 统一生成 embedding、构建 FAISS 索引并保存
    """
    # 获取当前这组 PDF 对应的索引路径
    index_path, metadata_path = get_multi_pdf_storage_paths(pdf_paths)

    # 如果索引已存在，则直接加载
    if faiss_index_exists(index_path, metadata_path):
        index, metadata = load_faiss_index(index_path, metadata_path)
        return index, metadata

    # 如果索引不存在，则从头构建
    all_chunks = []

    # 逐个处理每个 PDF
    for pdf_path in pdf_paths:
        # 1. 读取 PDF 内容
        pages = load_pdf(pdf_path)

        # 2. 过滤噪声页
        pages = filter_pages(pages)

        # 3. 切分 chunk
        chunks = split_text(pages)

        # 4. 合并到总 chunk 列表
        all_chunks.extend(chunks)

    # 5. 为所有 chunk 统一生成 embedding
    embedded_chunks = embed_chunks(all_chunks)

    # 6. 构建 FAISS 索引
    index, metadata = build_faiss_index(embedded_chunks)

    # 7. 保存索引和 metadata，方便下次复用
    save_faiss_index(index, metadata, index_path, metadata_path)

    return index, metadata


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
        pdf_path = os.path.join("data", pdf_name)

        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在: {pdf_name}"
            )

        pdf_paths.append(pdf_path)

    # 构建或加载知识库索引
    index, metadata = build_rag_pipeline(pdf_paths)

    # 调用你现有的 RAG 逻辑生成答案
    result = generate_answer(
        query=question,
        index=index,
        metadata=metadata,
        top_k=request.top_k
    )

    # 返回结果
    return {
        "question": result["query"],
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"]
    }