# api/main.py

import os
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

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
    VECTOR_STORE_BACKEND,
    CHROMA_PATH,
)

from modules.doc_registry import (
    compute_kb_id,
    file_sha256,
    get_document,
    register_documents,
)
from modules.pdf_loader import load_pdf
from modules.text_splitter import split_text
from modules.data_cleaner import filter_pages
from modules.embeddings import embed_chunks
from modules.hybrid_retriever import HybridRetriever
from modules.chroma_store import ChromaVectorStore
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
_retriever_cache: OrderedDict[str, HybridRetriever] = OrderedDict()
_RETRIEVER_CACHE_LIMIT = 8
_chroma_store: ChromaVectorStore | None = None


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


def _pipeline_fingerprint() -> str:
    return (
        f"{INDEX_SCHEMA_VERSION}:{VECTOR_STORE_BACKEND}:"
        f"{PDF_PARSE_BACKEND}:{CHUNK_SIZE}:{CHUNK_OVERLAP}"
    )


def _cache_retriever(cache_key: str, index: Any, metadata: List[Dict[str, Any]]) -> HybridRetriever:
    """
    Cache one BM25-bound retriever per selection, with a bounded LRU policy.
    """
    retriever = _retriever_cache.get(cache_key)
    if retriever is not None:
        _retriever_cache.move_to_end(cache_key)
        return retriever

    while len(_retriever_cache) >= _RETRIEVER_CACHE_LIMIT:
        _retriever_cache.popitem(last=False)

    retriever = HybridRetriever(index, metadata)
    _retriever_cache[cache_key] = retriever
    return retriever


def _get_chroma_store() -> ChromaVectorStore:
    global _chroma_store

    if _chroma_store is None:
        fingerprint = hashlib.md5(
            f"{get_current_embedding_model()}:{_pipeline_fingerprint()}".encode("utf-8")
        ).hexdigest()[:12]
        _chroma_store = ChromaVectorStore(
            path=CHROMA_PATH,
            collection_name=f"course_chunks_{fingerprint}",
        )

    return _chroma_store


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


def _build_faiss_pipeline(pdf_paths: List[str]):
    """
    Legacy full-selection FAISS pipeline, retained as a fallback backend.
    """
    register_documents(pdf_paths)

    kb_id = compute_kb_id(
        pdf_paths,
        embedding_backend=EMBEDDING_BACKEND,
        embedding_model=get_current_embedding_model(),
        index_schema_version=_pipeline_fingerprint(),
    )
    index_path, metadata_path = get_kb_storage_paths(kb_id)
    print(f"当前 FAISS 知识库 kb_id: {kb_id}")

    if faiss_index_exists(index_path, metadata_path):
        index, metadata = load_faiss_index(index_path, metadata_path)
    else:
        all_chunks = []
        next_chunk_id = 0

        for pdf_path in sorted(pdf_paths, key=lambda p: os.path.basename(p).lower()):
            pages = filter_pages(load_pdf(pdf_path))
            chunks = split_text(
                pages,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                backend=PDF_PARSE_BACKEND,
                start_chunk_id=next_chunk_id,
            )
            all_chunks.extend(chunks)
            next_chunk_id += len(chunks)

        embedded_chunks = embed_chunks(all_chunks)
        index, metadata = build_faiss_index(embedded_chunks)
        save_faiss_index(index, metadata, index_path, metadata_path)

    return index, metadata, kb_id, _cache_retriever(kb_id, index, metadata)


def _build_chroma_pipeline(pdf_paths: List[str]):
    """
    Build or update a persistent Chroma index incrementally.

    Unchanged files are detected by (source, sha256) and skipped. A changed
    file first has its old hash deleted, then only that file is re-parsed and
    re-embedded. Queries and BM25 are limited to the current file selection.
    """
    sorted_paths = sorted(pdf_paths, key=lambda p: os.path.basename(p).lower())
    current_records = {
        os.path.basename(path): {
            "path": path,
            "sha256": file_sha256(path),
        }
        for path in sorted_paths
    }

    store = _get_chroma_store()

    # Remove vectors for a filename whose content hash changed before the
    # registry row is overwritten. This also covers the case where the new
    # version yields no parseable chunks (then upsert never runs).
    for source, current in current_records.items():
        previous = get_document(source)
        if previous and previous["sha256"] != current["sha256"]:
            store.delete_document(previous["sha256"], source)

    register_documents(sorted_paths)

    for source, current in current_records.items():
        # 内容未变化则跳过解析与 embedding，直接复用已有向量
        if store.has_document(current["sha256"], source):
            continue

        pages = filter_pages(load_pdf(current["path"]))
        chunks = split_text(
            pages,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            backend=PDF_PARSE_BACKEND,
        )

        # 可能为 0：例如扫描版 PDF 全部被噪声过滤掉。
        # 跳过而不是抛错，避免单个坏文件让整次问答失败。
        if not chunks:
            print(f"[警告] {source} 未解析出可索引内容，已跳过（可能为扫描版 PDF）")
            continue

        embedded_chunks = embed_chunks(chunks)
        store.upsert_document(current["sha256"], source, embedded_chunks)

    doc_hashes = [record["sha256"] for record in current_records.values()]
    metadata = store.get_documents(doc_hashes)
    if not metadata:
        raise RuntimeError("Chroma 知识库为空，请检查 PDF 是否包含可解析文本")

    selection_id = compute_kb_id(
        sorted_paths,
        embedding_backend=EMBEDDING_BACKEND,
        embedding_model=get_current_embedding_model(),
        index_schema_version=_pipeline_fingerprint(),
    )
    index = store.for_documents(doc_hashes)

    print(f"当前 Chroma 选择集 kb_id: {selection_id}")
    return index, metadata, selection_id, _cache_retriever(selection_id, index, metadata)


def build_rag_pipeline(pdf_paths: List[str]):
    """
    Build the retrieval backend configured by VECTOR_STORE_BACKEND.
    """
    if VECTOR_STORE_BACKEND == "chroma":
        return _build_chroma_pipeline(pdf_paths)

    if VECTOR_STORE_BACKEND == "faiss":
        return _build_faiss_pipeline(pdf_paths)

    raise ValueError(f"不支持的 VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND}")


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
