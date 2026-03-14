"""
RAG 系统统一配置文件
"""

# ================================
# Embedding backend configuration
# ================================

# 选择 embedding backend
# 可选：
# "api"  -> 使用当前的 API embedding
# "hf"   -> 使用 HuggingFace 本地 embedding
EMBEDDING_BACKEND = "hf"

# HuggingFace embedding 模型名称
# 这个模型体积小、速度快，非常适合 RAG
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==============================
# 文本切分参数
# ==============================

# 每个 chunk 的最大字符数
CHUNK_SIZE = 700

# chunk 之间重叠字符
CHUNK_OVERLAP = 150


# ==============================
# 向量检索参数
# ==============================

# 向量检索召回数量
VECTOR_TOP_K = 8


# ==============================
# BM25 检索参数
# ==============================

# BM25 召回数量
BM25_TOP_K = 8


# ==============================
# Reranker 参数
# ==============================

# Reranker 最终保留数量
RERANK_TOP_K = 5


# ==============================
# 最终给 LLM 的上下文数量
# ==============================

FINAL_TOP_K = 12


# ==============================
# 模型配置
# ==============================

# embedding 模型
EMBEDDING_MODEL = "text-embedding-3-small"

# 对话模型
CHAT_MODEL = "deepseek-chat"

