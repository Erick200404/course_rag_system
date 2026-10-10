"""
RAG 系统统一配置文件
"""

import os

# ==============================
# PDF 解析与切分
# ==============================

# PDF 解析后端
# "markdown" -> 使用 pymupdf4llm 转 Markdown（保留标题层级、表格、阅读顺序，推荐）
# "text"     -> 纯文本逐页提取（旧行为，兜底方案）
PDF_PARSE_BACKEND = "markdown"

# 索引结构版本。检索算法或 metadata 结构变化时更新该值，
# 避免旧 FAISS 索引被新代码误加载。
INDEX_SCHEMA_VERSION = "faiss-cosine-v2"

# 向量存储后端：
# "chroma" -> Chroma 持久化向量库，支持按文档增量更新和 metadata 过滤
# "faiss"  -> 原有 FAISS + metadata.pkl 方案，保留为兜底
VECTOR_STORE_BACKEND = "chroma"

# Chroma 持久化目录
CHROMA_PATH = os.path.join("storage", "chroma")

# 每个 chunk 的最大字符数
CHUNK_SIZE = 700

# chunk 之间重叠字符
CHUNK_OVERLAP = 150

# ==============================
# Embedding 配置
# ==============================

# 选择 embedding backend
# 可选：
# "api"  -> 使用 OpenAI-compatible API embedding
# "hf"   -> 使用 HuggingFace 本地 embedding
EMBEDDING_BACKEND = "hf"

# HuggingFace embedding 模型名称
# 中文语料场景，bge-small-zh-v1.5 明显优于英文模型 all-MiniLM-L6-v2
HF_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# API embedding 模型
EMBEDDING_MODEL = "text-embedding-3-small"

# ==============================
# 检索参数
# ==============================

# 向量检索召回数量
VECTOR_TOP_K = 8

# BM25 召回数量
BM25_TOP_K = 8

# Hybrid Retrieval 合并后保留数量（送入 reranker）
FINAL_TOP_K = 12

# Reranker 最终保留数量
RERANK_TOP_K = 5

# ==============================
# LLM 配置
# ==============================

# 选择 LLM backend
# 可选：
# "api"    -> 使用 OpenAI-compatible 在线接口
# "ollama" -> 使用本地 Ollama
LLM_BACKEND = "ollama"

# 在线 API 模型名
CHAT_MODEL = "deepseek-chat"

# Ollama 本地配置
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "deepseek-r1:7b"
OLLAMA_TIMEOUT = 120
