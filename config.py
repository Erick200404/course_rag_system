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

# ==============================
# 扫描页 OCR
# ==============================

# 是否启用扫描页 OCR 路由（检测到扫描页时自动走 OCR 提取文本）
OCR_ENABLED = True

# OCR 后端：
# "rapidocr" -> RapidOCR（ONNXRuntime，纯 pip 安装，CPU 也能跑，中文效果好）
# "none"     -> 不做 OCR，扫描页直接跳过（旧行为）
OCR_BACKEND = "rapidocr"

# 扫描页判定阈值：页面文本密度（字符数 / 页面面积[平方英寸]）低于该值视为扫描页。
# 典型参考：A4 满页文本约 3~6，扫描页（无文字层）约 0。
SCAN_TEXT_DENSITY_THRESHOLD = 0.1

# ==============================
# 图片提取 + VLM 描述
# ==============================

# 是否启用 PDF 图片提取与 VLM 描述入库
# 课件中的图表往往承载关键知识点，纯文本解析完全丢失这部分内容
IMAGE_EXTRACT_ENABLED = True

# 图片提取后存放目录（按文档 hash 分目录）
IMAGE_STORE_PATH = os.path.join("storage", "images")

# 过滤小图：宽或高低于该像素数的图片忽略（图标/装饰线条）
IMAGE_MIN_SIZE = 128

# 是否调用视觉模型生成图片描述
VLM_ENABLED = True

# Ollama 视觉模型（需提前 ollama pull）
VLM_MODEL = "qwen2.5vl:7b"

# Ollama 服务地址（与 LLM 共用）
VLM_BASE_URL = "http://127.0.0.1:11434"

# ==============================
# RAG 编排（LangGraph）
# ==============================

# 编排模式：
# "simple" -> 直线流程（检索 → 重排 → 生成），无分支，适合单轮问答
# "crag"   -> CRAG 自纠错检索（检索 → 相关性评分 → 不相关则改写查询重试）
#             有分支与循环，LangGraph 的价值在这里才体现
RAG_FLOW = "crag"

# CRAG：检索结果不相关时最多改写重试几次（超过则走兜底回答）
MAX_RETRIEVE_RETRIES = 2

# 多轮对话：注入查询改写的最近对话轮数
MAX_HISTORY_TURNS = 3

# VLM 单张图片描述的超时（秒）
VLM_TIMEOUT = 120

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
