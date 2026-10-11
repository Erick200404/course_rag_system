# api/main.py

import os
import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List
from threading import RLock
from contextlib import nullcontext

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import (
    RERANK_TOP_K,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    HF_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INDEX_SCHEMA_VERSION,
    IMAGE_EXTRACT_ENABLED,
    PDF_PARSE_BACKEND,
    VECTOR_STORE_BACKEND,
    CHROMA_PATH,
    VLM_ENABLED,
    RAG_FLOW,
)

from modules.doc_registry import (
    compute_kb_id,
    file_sha256,
    get_document,
    register_documents,
    remove_document,
)
from modules.document_loader import (
    SUPPORTED_EXTENSIONS,
    is_supported,
    load_document,
)
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
from modules.rag_graph import run_crag
from modules.llm_backends import ModelServiceError
from modules import workspace_store as workspace


# 创建 FastAPI 应用
app = FastAPI(
    title="Course RAG API",
    description="课程资料智能问答系统 API",
    version="1.1.0"
)


@app.exception_handler(ModelServiceError)
async def model_service_error_handler(request: Request, exc: ModelServiceError):
    logging.getLogger(__name__).error("模型服务调用失败", exc_info=exc)
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


# ==============================
# HybridRetriever 实例缓存
# ==============================
# kb_id -> HybridRetriever
# 修复说明：旧版 BM25 使用模块级全局单例，切换知识库后不重建；
# 现在每个知识库持有独立的 retriever 实例，BM25 随之正确切换。
_retriever_cache: OrderedDict[str, HybridRetriever] = OrderedDict()
_RETRIEVER_CACHE_LIMIT = 8
_chroma_store: ChromaVectorStore | None = None
_course_stores: Dict[str, ChromaVectorStore] = {}
_course_locks: Dict[str, RLock] = {}
_locks_guard = RLock()


def course_lock(course_id: str) -> RLock:
    with _locks_guard:
        return _course_locks.setdefault(course_id, RLock())


# ==============================
# 请求数据结构
# ==============================
class ChatTurn(BaseModel):
    """一轮对话历史（用于多轮查询改写）。"""

    question: str = Field(default="", description="该轮的用户问题")
    answer: str = Field(default="", description="该轮的系统回答")


class ChatRequest(BaseModel):
    # 用户问题
    question: str = Field(..., description="用户输入的问题")
    session_id: str | None = Field(default=None, description="持久化问答会话编号")

    # 要使用的文档文件名列表（文件需已存在于 data 目录中）
    doc_names: List[str] | None = Field(
        default=None,
        description="参与问答的文档文件名列表",
    )

    # 旧字段，保留兼容：老版本前端仍会传 pdf_names
    pdf_names: List[str] | None = Field(
        default=None,
        description="已废弃，请使用 doc_names",
    )

    # 多轮对话历史（CRAG 模式下用于查询改写）
    history: List[ChatTurn] | None = Field(
        default=None,
        description="最近的对话历史，按时间正序",
    )

    # 最终送入大模型的片段数量
    top_k: int = Field(default=RERANK_TOP_K, ge=1, le=10, description="最终召回片段数量")


class CourseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class SessionCreate(BaseModel):
    title: str = Field(default="新会话", max_length=100)


class DocumentSelection(BaseModel):
    doc_names: List[str]


def workspace_call(function, *args):
    try:
        return function(*args)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/courses")
def courses():
    return workspace.list_courses()


@app.post("/courses", status_code=201)
def add_course(request: CourseCreate):
    return workspace_call(workspace.create_course, request.name)


@app.get("/courses/{course_id}/documents")
def course_documents(course_id: str):
    return workspace_call(workspace.list_course_documents, course_id)


@app.put("/courses/{course_id}/documents/{filename}")
async def upload_course_document(course_id: str, filename: str, request: Request):
    workspace_call(workspace.get_course, course_id)
    workspace_call(workspace.validate_filename, filename)
    if not is_supported(filename):
        raise HTTPException(status_code=400, detail="不支持的文件格式")
    # Stream the body with a bound instead of loading an unbounded request.
    content = bytearray()
    async for piece in request.stream():
        content.extend(piece)
        if len(content) > workspace.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件不能超过 200MB")
    from starlette.concurrency import run_in_threadpool

    def save():
        with course_lock(course_id):
            return workspace_call(workspace.save_course_document, course_id, filename, bytes(content))

    return await run_in_threadpool(save)


@app.get("/courses/{course_id}/sessions")
def course_sessions(course_id: str):
    return workspace_call(workspace.list_sessions, course_id)


@app.post("/courses/{course_id}/sessions", status_code=201)
def add_session(course_id: str, request: SessionCreate):
    with course_lock(course_id):
        return workspace_call(workspace.create_session, course_id, request.title)


@app.get("/sessions/{session_id}")
def session_detail(session_id: str):
    return workspace_call(workspace.get_session, session_id)


@app.patch("/sessions/{session_id}/documents")
def session_documents(session_id: str, request: DocumentSelection):
    session = workspace_call(workspace.get_session, session_id)
    with course_lock(session["course_id"]):
        workspace_call(workspace.select_session_documents, session_id, request.doc_names)
    return {"status": "ok"}


@app.delete("/sessions/{session_id}/turns")
def clear_session_history(session_id: str):
    session = workspace_call(workspace.get_session, session_id)
    with course_lock(session["course_id"]):
        workspace.clear_turns(session_id)
    return {"status": "ok"}


def _remove_course_indexes(course_id: str, source: str | None = None):
    """Remove from every embedding/config collection, not just the active one."""
    storage = workspace.course_storage_dir(course_id)
    chroma_path = storage / "chroma"
    if chroma_path.is_dir():
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        for collection in client.list_collections():
            name = collection if isinstance(collection, str) else collection.name
            if source is None:
                client.delete_collection(name)
            else:
                client.get_collection(name).delete(where={"source": {"$eq": source}})
    # FAISS is a full-selection cache; discard it and rebuild retained files
    # on demand. A binary index cannot remove one file's metadata independently.
    if storage.is_dir():
        for directory in storage.iterdir():
            if directory.is_dir() and len(directory.name) == 32 and all(c in "0123456789abcdef" for c in directory.name):
                workspace.remove_owned_tree(storage, directory)
    registry = storage / "registry.db"
    if registry.is_file():
        if source is None:
            registry.unlink()
        else:
            remove_document(source, db_path=str(registry))
    _course_stores.pop(course_id, None)
    for key in list(_retriever_cache):
        if key.startswith(f"{course_id}:"):
            _retriever_cache.pop(key, None)


@app.delete("/courses/{course_id}/documents/{filename}")
def delete_document(course_id: str, filename: str):
    with course_lock(course_id):
        documents = workspace_call(workspace.list_course_documents, course_id)
        workspace_call(workspace.validate_filename, filename)
        if filename not in {document["name"] for document in documents}:
            raise HTTPException(status_code=404, detail="资料不存在")
        _remove_course_indexes(course_id, source=filename)
        workspace_call(workspace.delete_course_document, course_id, filename)
    return {"status": "ok"}


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    session = workspace_call(workspace.get_session, session_id)
    with course_lock(session["course_id"]):
        workspace_call(workspace.delete_session, session_id)
    return {"status": "ok"}


@app.delete("/courses/{course_id}")
def remove_course(course_id: str):
    with course_lock(course_id):
        workspace_call(workspace.get_course, course_id)
        _remove_course_indexes(course_id)
        workspace_call(workspace.delete_course, course_id)
    return {"status": "ok"}


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


def _get_chroma_store(course_id: str | None = None) -> ChromaVectorStore:
    global _chroma_store

    if course_id is not None:
        if course_id not in _course_stores:
            fingerprint = hashlib.md5(
                f"{EMBEDDING_BACKEND}:{get_current_embedding_model()}:{_pipeline_fingerprint()}".encode("utf-8")
            ).hexdigest()[:12]
            _course_stores[course_id] = ChromaVectorStore(
                path=str(workspace.course_storage_dir(course_id) / "chroma"),
                collection_name=f"course_chunks_{fingerprint}",
            )
        return _course_stores[course_id]

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


def _build_faiss_pipeline(doc_paths: List[str], course_id: str | None = None):
    """
    Legacy full-selection FAISS pipeline, retained as a fallback backend.
    """
    registry = str(workspace.course_storage_dir(course_id) / "registry.db") if course_id else None
    register_documents(doc_paths, db_path=registry)

    kb_id = compute_kb_id(
        doc_paths,
        embedding_backend=EMBEDDING_BACKEND,
        embedding_model=get_current_embedding_model(),
        index_schema_version=_pipeline_fingerprint(),
    )
    index_path, metadata_path = get_kb_storage_paths(kb_id)
    if course_id:
        directory = workspace.course_storage_dir(course_id) / kb_id
        index_path = str(directory / "faiss_index.bin")
        metadata_path = str(directory / "metadata.pkl")
        kb_id = f"{course_id}:{kb_id}"
    print(f"当前 FAISS 知识库 kb_id: {kb_id}")

    if faiss_index_exists(index_path, metadata_path):
        index, metadata = load_faiss_index(index_path, metadata_path)
    else:
        all_chunks = []
        next_chunk_id = 0

        for doc_path in sorted(doc_paths, key=lambda p: os.path.basename(p).lower()):
            pages = filter_pages(load_document(doc_path))
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


def _build_image_chunks(doc_path: str, doc_hash: str, source: str) -> List[Dict[str, Any]]:
    """
    PDF 图片提取 + VLM 描述，返回可入库的图片 chunk 列表。

    仅对 PDF 文件生效；提取/描述失败或未启用时返回空列表，不影响文本入库。
    """
    if not str(doc_path).lower().endswith(".pdf"):
        return []

    if not IMAGE_EXTRACT_ENABLED:
        return []

    try:
        from modules.image_extractor import extract_document_images
        images = extract_document_images(doc_path, doc_hash)
    except Exception as e:
        print(f"[警告] {source} 图片提取失败，跳过图片入库：{e}")
        return []

    if not images:
        return []

    print(f"[图片] {source}：提取到 {len(images)} 张图片")

    if not VLM_ENABLED:
        return []

    try:
        from modules.vlm_captioner import caption_images
        captioned = caption_images(images, source)
    except Exception as e:
        print(f"[警告] {source} 图片描述生成失败，跳过图片入库：{e}")
        return []

    from modules.image_extractor import build_image_chunk

    return [build_image_chunk(image, image["caption"], source) for image in captioned]


def _build_chroma_pipeline(doc_paths: List[str], course_id: str | None = None):
    """
    Build or update a persistent Chroma index incrementally.

    Unchanged files are detected by (source, sha256) and skipped. A changed
    file first has its old hash deleted, then only that file is re-parsed and
    re-embedded. Queries and BM25 are limited to the current file selection.
    """
    sorted_paths = sorted(doc_paths, key=lambda p: os.path.basename(p).lower())
    current_records = {
        os.path.basename(path): {
            "path": path,
            "sha256": file_sha256(path),
        }
        for path in sorted_paths
    }

    store = _get_chroma_store(course_id) if course_id else _get_chroma_store()
    registry = str(workspace.course_storage_dir(course_id) / "registry.db") if course_id else None

    # Remove vectors for a filename whose content hash changed before the
    # registry row is overwritten. This also covers the case where the new
    # version yields no parseable chunks (then upsert never runs).
    for source, current in current_records.items():
        previous = get_document(source, db_path=registry)
        if previous and previous["sha256"] != current["sha256"]:
            store.delete_document(previous["sha256"], source)

    register_documents(sorted_paths, db_path=registry)

    for source, current in current_records.items():
        # 内容未变化则跳过解析与 embedding，直接复用已有向量
        if store.has_document(current["sha256"], source):
            continue

        pages = filter_pages(load_document(current["path"]))
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

        # PDF 图片提取 + VLM 描述，生成可检索的图片 chunk
        chunks.extend(_build_image_chunks(current["path"], current["sha256"], source))

        if not chunks:
            continue

        embedded_chunks = embed_chunks(chunks)
        store.upsert_document(current["sha256"], source, embedded_chunks)

    doc_hashes = [record["sha256"] for record in current_records.values()]
    sources = list(current_records)
    metadata = store.get_documents(doc_hashes, sources=sources)
    if not metadata:
        raise RuntimeError("Chroma 知识库为空，请检查文档是否包含可解析文本")

    selection_id = compute_kb_id(
        sorted_paths,
        embedding_backend=EMBEDDING_BACKEND,
        embedding_model=get_current_embedding_model(),
        index_schema_version=_pipeline_fingerprint(),
    )
    index = store.for_documents(doc_hashes, sources=sources)
    if course_id:
        selection_id = f"{course_id}:{selection_id}"

    print(f"当前 Chroma 选择集 kb_id: {selection_id}")
    return index, metadata, selection_id, _cache_retriever(selection_id, index, metadata)


def build_rag_pipeline(doc_paths: List[str], course_id: str | None = None):
    """
    Build the retrieval backend configured by VECTOR_STORE_BACKEND.
    """
    if VECTOR_STORE_BACKEND == "chroma":
        return _build_chroma_pipeline(doc_paths, course_id=course_id)

    if VECTOR_STORE_BACKEND == "faiss":
        return _build_faiss_pipeline(doc_paths, course_id=course_id)

    raise ValueError(f"不支持的 VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND}")


# ==============================
# 问答接口
# ==============================
@app.post("/chat")
def chat(request: ChatRequest):
    session = workspace_call(workspace.get_session, request.session_id) if request.session_id else None
    # Serialize writes and retrieval within a course: concurrent uploads cannot
    # replace files while their old vectors are being queried or rebuilt.
    with course_lock(session["course_id"]) if session else nullcontext():
        return _chat(request)


def _chat(request: ChatRequest):
    """
    根据指定文档文件和问题，返回 RAG 问答结果。
    """
    # 去掉问题首尾空格
    question = request.question.strip()

    # 检查问题是否为空
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    # doc_names 为主字段，pdf_names 仅为旧版前端兼容
    session = workspace_call(workspace.get_session, request.session_id) if request.session_id else None
    doc_names = request.doc_names if request.doc_names is not None else request.pdf_names
    if session and doc_names is None:
        doc_names = session["selected_documents"]
        if doc_names is None:
            doc_names = [doc["name"] for doc in workspace.list_course_documents(session["course_id"])]

    # 检查文档文件名列表是否为空
    if not doc_names:
        raise HTTPException(status_code=400, detail="doc_names 不能为空")

    # 将文件名转为 data 目录下的路径
    doc_paths = []
    for doc_name in doc_names:
        workspace_call(workspace.validate_filename, doc_name)

        if not is_supported(doc_name):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {doc_name}，"
                       f"当前支持 {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        doc_path = (workspace_call(workspace.resolve_documents, session["course_id"], [doc_name])[0]
                    if session else os.path.join("data", doc_name))

        # 检查文件是否存在
        if not os.path.exists(doc_path):
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在: {doc_name}"
            )

        doc_paths.append(doc_path)

    # 构建或加载知识库索引
    if session:
        workspace.select_session_documents(session["id"], doc_names)
        index, metadata, kb_id, retriever = build_rag_pipeline(doc_paths, course_id=session["course_id"])
        # Server-owned history prevents one session from borrowing another's context.
        history = [{"question": turn["question"], "answer": turn["result"]["answer"]}
                   for turn in session["turns"]]
    else:
        index, metadata, kb_id, retriever = build_rag_pipeline(doc_paths)
        history = request.history

    if RAG_FLOW == "crag":
        # CRAG：检索 → 评分 → 不相关则改写重试（有分支与循环，交给 LangGraph 编排）
        result = run_crag(
            question=question,
            retriever=retriever,
            top_k=request.top_k,
            history=history,
        )

        response = {
            "question": result["question"],
            "original_question": result["original_question"],
            "answer": result["answer"],
            "retrieved_chunks": result["chunks"],
            "flow": "crag",
            "retries": result["retries"],
            "fallback": result["fallback"],
        }
        if session:
            workspace.append_turn(session["id"], question, response)
        return response

    if RAG_FLOW == "simple":
        # simple：直线流程，无分支，保持最小开销
        result = generate_answer(
            query=question,
            retriever=retriever,
            top_k=request.top_k
        )

        response = {
            "question": result["query"],
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
            "flow": "simple",
        }
        if session:
            workspace.append_turn(session["id"], question, response)
        return response

    raise ValueError(f"不支持的 RAG_FLOW: {RAG_FLOW}")
