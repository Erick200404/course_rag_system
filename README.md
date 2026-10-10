
# 基于 RAG 的课程 PDF 智能问答与复习辅助系统

## 一、项目简介

本项目实现了一个 **基于 Retrieval-Augmented Generation (RAG)** 的课程资料智能问答系统。

系统允许用户上传课程 PDF（课件、讲义、实验报告等），并通过自然语言提问，系统会自动检索相关内容并生成答案，同时给出 **引用来源（文件名 + 页码）**，帮助用户快速定位知识点。

该项目主要用于探索 **大语言模型 + 信息检索技术** 在学习辅助场景中的应用。  
系统整体采用：

```
Streamlit + FastAPI + RAG Pipeline
```

架构实现 **前端界面 + 后端服务 + AI检索生成** 的完整流程。

![系统示例图.png](doc/%E7%B3%BB%E7%BB%9F%E7%A4%BA%E4%BE%8B%E5%9B%BE.png)

------

## 二、项目背景

大学生在期末复习时通常需要反复翻阅老师提供的 PDF 课件、讲义或实验报告。

常见问题包括：

- 查找某个知识点需要翻阅大量 PDF
- 不容易快速定位答案所在页
- 多个课件之间难以统一检索

因此，本项目希望构建一个 **课程资料智能问答系统**，让用户可以通过提问直接获取答案，并快速跳转到原始资料页。

------

## 三、项目目标

实现一个支持以下功能的系统：

- 多 PDF 上传
- 文档内容解析与处理
- 基于语义的知识检索
- 结合大语言模型生成回答
- 自动标注答案引用来源
- 提供简单易用的 Web UI

------

# 系统架构

整体系统架构如下：

```
User
 ↓
Streamlit UI
 ↓
FastAPI
 ↓
RAG Pipeline
 ↓
PDF Loader
 ↓
Text Splitter
 ↓
Embedding
 ↓
Vector Store (FAISS)
 ↓
Hybrid Retrieval
 ↓
Reranker
 ↓
LLM
 ↓
Answer + Reference
```

------

# 技术栈

主要使用以下技术：

```
Python
FastAPI
Streamlit
FAISS
BM25
BGE Reranker
HuggingFace Embedding
OpenAI Compatible API
Ollama (Local LLM)
PyMuPDF
```

核心技术包括：

- Retrieval-Augmented Generation (RAG)
- Hybrid Retrieval（向量检索 + 关键词检索）
- Reranking
- 多文档知识库
- API 服务化架构

------

# 项目结构

```
course_rag_system
│
├─ api
│   └─ main.py            # FastAPI 服务接口
│
├─ modules
│   ├─ pdf_loader.py      # PDF 解析（markdown / text 双后端）
│   ├─ data_cleaner.py    # 噪声页过滤
│   ├─ text_splitter.py   # 标题层级感知切分 + 三级兜底
│   ├─ doc_registry.py    # SQLite 文档注册表（内容 hash）
│   ├─ embeddings.py      # 双后端 embedding（懒加载）
│   ├─ chroma_store.py    # Chroma 持久化向量库（增量更新）
│   ├─ vector_store.py    # FAISS 索引（兜底后端）
│   ├─ bm25_retriever.py
│   ├─ hybrid_retriever.py
│   ├─ reranker.py
│   └─ rag_chain.py
│
├─ tests                  # pytest 单元测试
├─ conftest.py
│
├─ data                   # PDF 文件
├─ storage                # 向量索引缓存 + registry.db
│
├─ app.py                 # Streamlit 前端
├─ config.py              # 系统配置
├─ requirements.txt
└─ README.md
```

------

# 快速启动

## 1 环境准备

建议使用 **conda 环境**

```
conda create -n course_rag python=3.10
conda activate course_rag
```

安装依赖：

```
pip install -r requirements.txt
```

------

## 2 配置文件

在项目根目录创建 `data` 和 `storage` 目录。  

在项目根目录创建 `.env` 文件并按照如下示例填写：

```
ZHI_API_KEY=你的APIKEY
ZHI_BASE_URL=你的API接口地址
```

------

## 2.1 本地模型（可选）

系统支持使用 **Ollama 本地大模型** 进行推理。

安装 Ollama：

https://ollama.com

下载模型：

```
ollama pull deepseek-r1:7b
```

启动 Ollama 服务：

```
ollama serve
```

然后在 `config.py` 中修改：

```
LLM_BACKEND = "ollama"
OLLAMA_MODEL = "deepseek-r1:7b"
```

即可使用本地模型进行问答。

------

## 3 启动系统

进入项目目录：

```
cd E:\vscode_workspace\course_rag_system
```

启动 FastAPI

```
uvicorn api.main:app --reload
```

打开 **另一个终端** 启动 Streamlit：

```
streamlit run app.py
```

浏览器打开：

```
http://localhost:8501
```

即可使用系统。

------

# Roadmap（开发历程）

本项目按照多个版本逐步迭代开发。

------

# v1.1.0 基础 RAG 系统

实现最基础的 RAG pipeline：

- [x] PDF Loader
- [x] Text Splitter
- [x] Embedding 模块
- [x] Vector Store（FAISS）
- [x] Retriever
- [x] RAG Pipeline
- [x] Streamlit UI

系统可以：

- 上传 PDF
- 提问
- 检索相关内容
- 生成回答

------

# v1.2.0 检索能力优化

提升检索质量。

新增：

- [x] Vector Store 数据持久化
- [x] Hybrid Retrieval（BM25 + Vector）
- [x] Reranker 重排序

改进：

- 检索召回质量明显提升
- 回答相关性提高

------

# v1.3.0 工程结构优化

对系统结构进行优化：

- [x] 优化 Text Splitter
- [x] 引入统一 `config` 管理参数
- [x] BM25 中文分词优化
- [x] 检索参数可配置化

------

# v1.4.0 多文档知识库

支持多 PDF 共同组成知识库：

- [x] 多 PDF 上传
- [x] 联合向量索引
- [x] 跨文档检索
- [x] 引用来源展示（文件名 + 页码）

系统从：

```
单文档问答
```

升级为：

```
多文档知识库问答
```

------

# v2.0.0 工程化升级

新增系统工程化能力：

- [x] 引入 FastAPI 构建问答服务接口
- [x] 新增 `/chat` RAG 问答 API
- [x] 新增 `/health` 服务健康检查接口
- [x] Streamlit 前端通过 HTTP 调用 FastAPI
- [x] 支持 HuggingFace 本地 embedding 模型
- [x] 自动检测 CUDA 并使用 GPU 推理
- [x] 修复 embedding 模型切换导致的索引冲突问题

系统架构从：

```
Streamlit + RAG
```

升级为：

```
Streamlit
 ↓
FastAPI
 ↓
RAG Pipeline
```

提升系统工程化程度。

------

# v2.1.0 本地模型支持

新增本地 LLM 推理能力：

- [x] 支持 Ollama 本地模型推理
- [x] 新增 `LLM_BACKEND` 配置
- [x] 支持 API 模型与本地模型自由切换
- [x] 本地模型使用 `deepseek-r1:7b`
- [x] 优化本地推理稳定性（使用 `/api/generate` 接口）

系统 LLM 调用链升级为：

```
LLM Backend
 ├─ API (deepseek-chat)
 └─ Ollama (deepseek-r1)
```

实现 **在线模型与本地模型统一适配**。

------

# v3.0.0 检索质量与工程化重构

聚焦底层修复与检索质量升级：

- [x] 修复 BM25 全局单例 bug：重构为 `HybridRetriever` 类，BM25 生命周期与知识库实例绑定，切换知识库后正确重建
- [x] 新增 SQLite 文档注册表（`modules/doc_registry.py`）：kb_id 改由 **文件内容 sha256 + embedding 配置** 共同决定，文件更新 / 换模型后自动重建索引，解决旧索引复用问题
- [x] embedding 换为中文模型 `BAAI/bge-small-zh-v1.5`，支持批量编码与 BGE 查询指令前缀
- [x] PDF 解析升级为 `pymupdf4llm`（Markdown 结构，保留标题层级 / 表格 / 阅读顺序），切分改为 **标题层级感知**，chunk 携带章节路径（标题增强）
- [x] Reranker 懒加载 + CUDA 自动检测
- [x] 新增 pytest 单元测试（`tests/`，35 个用例）

------

# v3.1.0 索引正确性修复

聚焦多文档索引与溯源 metadata 的正确性：

- [x] 修复同名 PDF 上传后不覆盖旧文件的问题：改用临时文件 + 原子替换，确保内容 hash 基于最新文件计算
- [x] 修复 FAISS metadata 丢失章节路径的问题：持久化 embedding 之外的完整 chunk 字段
- [x] 标题层级跨页延续：章节标题不再因分页被重置，后续页面 chunk 继续携带章节路径
- [x] 向量检索从 L2 距离升级为 cosine similarity：文档与查询向量统一归一化，FAISS 使用 `IndexFlatIP`
- [x] 多 PDF 建库时生成全局唯一 `chunk_id`，并按文件名稳定排序保证索引确定性
- [x] 引入索引结构版本号：检索算法、解析后端或切分参数变化时自动生成新的 `kb_id`，避免误加载旧索引
- [x] 上传与 API 入参增加文件名路径校验，拒绝路径穿越
- [x] 单元测试扩展至 44 个用例

# v3.2.0 Chroma 向量数据库化

把向量存储从「FAISS + pickle 整库重建」升级为可持久化、可增量维护的 Chroma 集合：

- [x] 新增 `modules/chroma_store.py`：持久化集合（cosine 空间），chunk 的向量 / 文本 / metadata 统一存储，支持按 `doc_hash` 与 `source` 过滤
- [x] `api/main.py` 双后端路由：`VECTOR_STORE_BACKEND` 可切换 `chroma` / `faiss`（FAISS 保留为兜底）
- [x] 增量建库：按 `(source, sha256)` 判断文件是否变化，未变化的文件跳过解析与 embedding；内容变化时清除旧版本向量再写入新版本
- [x] 检索选择集隔离：查询与 BM25 都限定在当前请求选中的文件（ChromaVectorView + metadata 过滤）
- [x] 集合名按「embedding 模型 + 索引结构版本 + 解析后端 + 切分参数」生成指纹，避免不同配置复用同一集合
- [x] 稳定性修复：`upsert_document` 改为按 source 整体替换（旧版本向量不再残留）；单文件解析不出 chunk 时跳过而非抛错；`rag_chain` 的 OpenAI 客户端改为懒加载，未配置 `.env` 也能启动 Ollama 后端
- [x] 单元测试扩展至 54 个用例（新增 Chroma store 单元测试与 `_build_chroma_pipeline` 集成测试）

配置示例：

```python
# config.py
VECTOR_STORE_BACKEND = "chroma"   # "chroma" | "faiss"
CHROMA_PATH = os.path.join("storage", "chroma")
INDEX_SCHEMA_VERSION = "faiss-cosine-v2"
```

规划中（v3.x）：

- [ ] LangChain 基础设施（loaders / splitters / LCEL）
- [ ] 扫描页 OCR 路由（按页文本密度检测，PaddleOCR / MinerU）
- [ ] 图片提取 + VLM 描述入库（qwen2.5vl via Ollama）
- [ ] LangGraph 编排（Query Rewrite / 多轮对话 / CRAG 自纠错检索）

------

# 模块说明

## pdf_loader.py

负责：

- 解析 PDF
- 提取每一页文本
- 保存页码和来源信息

当前限制：

- 仅支持文本型 PDF
- 扫描版 PDF 需要 OCR 才能处理

------

## text_splitter.py

负责：

- 将 PDF 页面切分为多个 chunk
- Markdown 后端按标题层级切分，并跨页维护章节路径
- chunk 携带章节路径，并对 embedding 文本做标题增强
- 长区段依次回退到段落、句子、字符三级切分，并保留 overlap
- 支持多文档合并时的全局 `chunk_id`

------

## embeddings.py

负责：

- 调用 embedding API
- 生成文本向量

当前使用：

```
API 后端：text-embedding-3-small
HF 后端：BAAI/bge-small-zh-v1.5
```

系统会根据配置自动选择 embedding backend，并支持 GPU 加速。

------

## vector_store.py

负责：

- 构建归一化 FAISS 向量索引
- 保存 metadata
- 支持索引持久化

功能：

- 向量存储
- cosine similarity 检索
- 持久化章节路径等完整 chunk metadata

------

## chroma_store.py

负责（默认后端）：

- 持久化 Chroma 集合，向量 / 文本 / metadata 统一存储（cosine 空间）
- 按 `(source, doc_hash)` 写入与删除，支持单文档整体替换
- `get_documents` / `query` 支持按 `doc_hash` 过滤，实现「当前选择的文件」隔离

与 vector_store.py 的区别：

| | Chroma（默认） | FAISS（兜底） |
| --- | --- | --- |
| 更新粒度 | 单文件增量 | 整个知识库重建 |
| metadata | 存在集合内 | 与向量分开存 pickle |
| 过滤检索 | 原生支持 | 需自行实现 |

------

## bm25_retriever.py

实现传统关键词检索：

- BM25 算法
- 中文分词
- 与向量检索结合

------

## hybrid_retriever.py

实现 **Hybrid Retrieval**：

```
Vector Retrieval
+
BM25 Retrieval
```

提升召回率。

------

## reranker.py

使用 **BGE Reranker** 对召回结果进行重排序。

作用：

- 提升最终答案相关性
- 过滤噪声 chunk

------

## rag_chain.py

负责完整 RAG 流程：

```
Query
 ↓
Hybrid Retrieval
 ↓
Reranker
 ↓
Context
 ↓
LLM
 ↓
Answer
```

LLM 调用支持多 backend：

API 模型：

```
deepseek-chat
POST /v1/chat/completions
```

本地模型：

```
deepseek-r1:7b
POST /api/generate
```

同时生成：

```
引用来源（文件 + 页码）
```

------

## app.py

使用 **Streamlit** 构建前端界面。

支持：

- 上传 PDF
- 输入问题
- 显示回答
- 显示引用来源
- 显示召回片段

前端通过 HTTP 调用 FastAPI `/chat` 接口完成问答。

------

# 未来优化方向

未来可以继续优化：

- 语义级文本切分
- Query Rewrite
- Answer Evaluation
- 更强的本地 LLM（Qwen / Llama / DeepSeek）
- 向量数据库（Milvus / Qdrant）
- 支持更多文档格式（docx、ppt）

------

# 项目总结

本项目实现了一个 **完整的 RAG 问答系统原型**，包括：

- 文档解析
- 知识检索
- 检索优化
- 多文档知识库
- 问答生成
- 引用溯源

并支持 **在线大模型 API 与本地大模型（Ollama）两种推理模式**。

适用于：

- 课程资料问答
- 学习辅助
- 小型知识库问答系统
