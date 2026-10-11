
# 基于 RAG 的课程 PDF 智能问答与复习辅助系统

## 当前使用方式：课程资料库与持久化会话

系统按「课程 → 问答会话」组织资料与对话：

1. 在侧栏创建课程，例如「机器学习」「数据库」。
2. 上传该课程的课件或笔记；文件上传后立即由 FastAPI 保存，不需要先提问。
3. 新建问答会话，选择该会话使用的资料。一个课程可包含多个独立会话。
4. 提问时按需建立或更新检索索引；后端保存问题、答案、引用依据及检索结果。
5. 重新打开页面，可选择已有课程和会话继续讨论；URL 中的课程与会话 ID 可用于恢复当前页面。

不同课程的同名文件、注册表和向量索引彼此隔离。同一课程的会话共用资料库，分别保存资料选择与历史。
「清空当前会话」只删除该会话的问答记录，不删除课程资料，也不影响其他会话。

删除操作在侧栏的管理菜单中，均需点击明确的确认按钮：

- 「管理课程」：删除当前课程、其原始资料、所有会话和检索索引。
- 「管理已上传资料」：选择并删除一份资料，同时清理所有配置的 Chroma 集合中的对应向量、
  文档注册记录、FAISS 缓存和 BM25 实例缓存，并从该课程各会话的资料选择中移除。
  历史回答保留；引用标记为「资料已删除」，原图不再展示。上传框会重置，避免自动重新上传。
- 「管理当前会话」：可清空历史，或删除整个会话；课程资料不受影响。

共享的图片提取缓存按文档 hash 保存，删除资料不会直接递归删除共享图片目录。
Chroma 集合内容会清理，数据库目录可能保留引擎内部结构。

持久化布局（路径相对于启动项目的工作目录）：

```text
data/courses/<course_id>/<filename>       原始课程资料
storage/workspace.db                     课程、资料目录、会话、资料选择、完整问答结果
storage/courses/<course_id>/registry.db  内容哈希注册表
storage/courses/<course_id>/chroma/      Chroma 持久化索引（默认）
storage/courses/<course_id>/<kb_id>/     FAISS 索引与 metadata（备用配置）
storage/images/<doc_hash>/              PDF 提取图片
```

前端负责交互，文件上传、课程管理、会话历史和检索均由后端负责。Streamlit 的 session_state
仅保存临时界面状态；重新打开页面时以 SQLite 中的数据为准。当前面向个人单用户使用。
旧版 `POST /chat` 的 `doc_names` / `pdf_names` 调用方式继续兼容；新版前端传 `session_id`，
由后端读取对应课程与历史。旧版仅保存在浏览器 session_state 的聊天记录无法在会话关闭后恢复。
旧版 `data/` 根目录的文件仍保留，可重新上传到指定课程建立新的课程归属。

主要新增接口：

| 接口 | 用途 |
| --- | --- |
| `GET /courses` / `POST /courses` | 查看、创建课程 |
| `GET /courses/{id}/documents` | 查看课程已有资料 |
| `PUT /courses/{id}/documents/{filename}` | 原始二进制文件上传（立即保存） |
| `GET /courses/{id}/sessions` / `POST /courses/{id}/sessions` | 查看、创建会话 |
| `GET /sessions/{id}` | 恢复完整会话 |
| `PATCH /sessions/{id}/documents` | 保存会话资料选择 |
| `DELETE /sessions/{id}/turns` | 清空当前会话历史 |
| `DELETE /sessions/{id}` | 删除会话与其历史 |
| `DELETE /courses/{id}/documents/{filename}` | 删除课程资料并清理索引 |
| `DELETE /courses/{id}` | 删除课程、资料和全部会话 |

前端使用 Streamlit 1.65+ 的原生会话组件与暖纸主题。顶部原生侧栏展开入口保持可见，
收起侧栏后可通过左上角按钮重新打开。修改后需重启前后端以加载新接口和主题。

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
Chroma（默认向量库，FAISS 兜底）
FAISS
BM25
BGE Reranker
HuggingFace Embedding
LangChain LCEL（生成链编排）
LangGraph（CRAG 自纠错检索编排）
OpenAI Compatible API
Ollama (Local LLM)
PyMuPDF / pymupdf4llm
RapidOCR（扫描页）
```

核心技术包括：

- Retrieval-Augmented Generation (RAG)
- Hybrid Retrieval（向量检索 + 关键词检索）
- Reranking
- 多文档知识库
- CRAG 自纠错检索（LangGraph 编排：相关性评分 + 查询改写重试）
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
│   ├─ document_loader.py # 多格式接入（pdf/docx/pptx/md/txt）
│   ├─ llm_backends.py    # LangChain LCEL 生成链
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
│   ├─ rag_chain.py
│   └─ rag_graph.py         # LangGraph CRAG 自纠错检索编排
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
conda create -n course_rag python=3.11
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

# v3.6.0 LangGraph 编排（CRAG 自纠错检索）

在前几版「直线 RAG 流程」基础上，引入 **LangGraph** 处理真正存在分支与循环的检索场景——**Corrective RAG（CRAG）**：

- [x] 新增 `modules/rag_graph.py`：用 `StateGraph` 组装 CRAG 状态机
      ```
      retrieve → grade →[relevant]→ generate → END
                       →[irrelevant & 重试未耗尽]→ rewrite → retrieve（循环）
                       →[irrelevant & 重试耗尽]→ fallback → END
      ```
      - `retrieve`：混合检索 + 重排（复用 `HybridRetriever` / `rerank_chunks`）
      - `grade`：让模型判断召回内容能否回答（仅输出 relevant / irrelevant）
      - `rewrite`：结合多轮历史把指代补全成可独立检索的完整问题
      - `generate` / `fallback`：复用 `rag_chain.build_prompt` 与 LCEL 链；重试耗尽仍不相关时诚实告知，不编造
- [x] **关键修复**：检索完全为空（知识库里根本没有相关内容）时直接走兜底，不再无意义地改写重试，避免浪费 LLM 调用与等待
- [x] 多轮对话：通过 `MAX_HISTORY_TURNS` 注入最近对话，支持指代消解（"它" / "这个方法"）
- [x] 编排模式可切换：`config.RAG_FLOW = "crag"` 走 CRAG；`"simple"` 退回直线流程（无分支场景保持最小开销）。框架只在「有分支/循环」的地方才引入，符合「框架只用在真正有收益处」的取舍
- [x] `api/main.py` 与 `app.py` 接入 CRAG：响应新增 `flow` / `retries` / `fallback` / `original_question` 字段，前端展示改写过程与兜底提示
- [x] 配置项：`RAG_FLOW` / `MAX_RETRIEVE_RETRIES` / `MAX_HISTORY_TURNS`（config.py）
- [x] 新增 `tests/test_rag_graph.py`（14 个用例：节点、条件边、空检索兜底、改写循环上限、多轮历史）与 `tests/test_api_flow.py`（4 个用例：crag/simple 分派、未知模式、历史透传）
- [x] 单元测试扩展至 112 个用例

------

# v3.5.0 图片提取 + VLM 描述入库

课件中的图表（架构图 / 流程图 / 公式截图）承载大量知识点，纯文本解析完全丢失。本版本把图片变成可检索的内容：

- [x] 新增 `modules/image_extractor.py`：PDF 内嵌图片提取
      - 按 xref 去重（同一图片多页引用只提取一次），保存到 `storage/images/<doc_hash>/`
      - 尺寸过滤（默认 128px 以下忽略图标/装饰），重复提取复用已保存文件
- [x] 新增 `modules/vlm_captioner.py`：调用 **Ollama 视觉模型**（默认 `qwen2.5vl:7b`，`/api/chat` + base64）生成中文描述
      - 提示词约束输出为客观的图表知识点描述（图表类型 / 核心信息 / 公式转写）
      - 降级策略：Ollama 不可达立即停止剩余图片；单张失败跳过；均不影响文本入库
- [x] 图片 chunk 入库：描述文本作为 chunk 参与检索，metadata 携带 `image_path`，前端召回时直接展示原图
- [x] 配置项：`IMAGE_EXTRACT_ENABLED` / `IMAGE_MIN_SIZE` / `VLM_ENABLED` / `VLM_MODEL` / `VLM_TIMEOUT`
- [x] 单元测试扩展至 94 个用例（新增图片提取 / VLM 伪造调用 / 端到端入库与检索 12 个）
- [x] 冒烟验证：图片提取（含小图过滤）与 Ollama 离线降级均符合预期；真实 VLM 描述需本机 `ollama pull qwen2.5vl:7b` 后启用

------

# v3.4.0 扫描页 OCR 路由

- [x] 新增 `modules/ocr_router.py`：扫描页自动检测与 OCR 提取
      - 检测：按「文本密度 = 字符数 / 页面面积（平方英寸）」判定，正常文本页约 1~6，扫描页接近 0
      - 路由：`text` 后端检测到扫描页时整册走 OCR；`markdown` 后端对空文本页按页码对齐补齐
      - 后端可插拔：默认 **RapidOCR**（ONNXRuntime，纯 pip，CPU 可跑）；`OCR_BACKEND="none"` 退回旧行为
      - 引擎懒加载，未启用时不下载模型；OCR 失败/未安装时降级为文字层提取，不中断解析
- [x] 配置项：`OCR_ENABLED` / `OCR_BACKEND` / `SCAN_TEXT_DENSITY_THRESHOLD`（config.py）
- [x] 单元测试扩展至 82 个用例（新增 OCR 检测/路由/降级/混合页 10 个，OCR 引擎用 monkeypatch 伪造）
- [x] 真实 RapidOCR 冒烟验证：中英文扫描页均可正确识别（首次运行自动下载内置模型）

------

# v3.3.0 多格式接入与 LangChain LCEL 编排

- [x] 新增 `modules/document_loader.py`：统一文档接入层，支持 **PDF / Word(docx) / PPT(pptx) / Markdown / TXT**
      - PPT 按幻灯片映射为「页」，使引用溯源对课件同样有效
      - Word 同时提取段落与表格文本
      - API 请求字段由 `pdf_names` 调整为 `doc_names`（保留旧字段兼容）
- [x] 新增 `modules/llm_backends.py`：用 **LangChain LCEL** 组装生成链
      （`ChatPromptTemplate | chat model | StrOutputParser`），替代手写的 OpenAI SDK 与
      Ollama `requests` 调用，两个后端只差一个 model 对象
- [x] 技术选型说明：解析层刻意不用 `langchain-community` 的 loader（该包已进入 sunset，
      导入会打印弃用警告），改用 python-docx / python-pptx 等轻量库；
      LangChain 用在真正有收益的编排层
- [x] 单元测试扩展至 72 个用例（新增文档接入 10 个、LCEL 链 8 个）

**注意**：Ollama 后端由原来的 `/api/generate` 改为 LangChain `ChatOllama` 的 `/api/chat`，
system / user 角色通过 ChatPromptTemplate 正确传递，`temperature` 与 `num_ctx` 参数保持不变。

------

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

原定路线图已全部完成（v3.3 多格式接入 + LCEL、v3.4 扫描页 OCR 路由、v3.5 图片提取 + VLM 描述入库、v3.6 LangGraph CRAG 自纠错检索均已交付）。后续可探索方向见「未来优化方向」。

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

## document_loader.py

统一文档接入层，把不同格式归一化为 `{page, text, source}`：

| 格式 | 实现 | 「页」的含义 |
| --- | --- | --- |
| pdf | pymupdf4llm（自研后端） | PDF 页码 |
| pptx | python-pptx | 幻灯片序号 |
| docx | python-docx（段落 + 表格） | 整份文档记为第 1 页 |
| md / txt | 直接读取 | 整份文档记为第 1 页 |

## llm_backends.py

用 LangChain LCEL 组装生成链：

```
ChatPromptTemplate  ->  chat model  ->  StrOutputParser
   (system + human)     (ChatOpenAI / ChatOllama)
```

两个 LLM 后端只差一个 model 对象，后续要加流式输出、JSON 结构化输出或重试，
只需替换链中的某个环节。

------

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

负责「直线流程」RAG 编排（simple 模式），并抽取出被 CRAG 复用的公共逻辑：

```
Query
 ↓
Hybrid Retrieval
 ↓
Reranker
 ↓
Context（build_prompt 拼装）
 ↓
LLM（invoke_llm -> LCEL 链）
 ↓
Answer
```

- `generate_answer`：simple 模式入口，检索 → 重排 → 生成带引用的答案
- `build_prompt` / `build_context`：上下文拼装，被 `rag_graph.generate_node` 直接复用，保证两条路径的 prompt 一致
- LLM 调用统一走 `modules/llm_backends.invoke_llm`（LangChain LCEL，ChatOpenAI / ChatOllama），不再手写 HTTP 请求

同时生成：

```
引用来源（文件 + 页码 + 图片）
```

## rag_graph.py

LangGraph `StateGraph` 编排层（crag 模式），只在「存在分支与循环」的检索场景引入：

- 节点：`retrieve` / `grade` / `rewrite` / `generate` / `fallback`，状态在 `RagState` 中流转（question / history / chunks / relevant / retries / answer / fallback）
- 条件边 `route_after_grade`：相关则生成；不相关且未超 `MAX_RETRIEVE_RETRIES` 则改写重试；检索为空或重试耗尽则兜底
- 检索、重排、生成全部复用 `HybridRetriever` / `rerank_chunks` / `rag_chain` 的现有实现，本模块只做编排，不重复造轮子
- 多轮对话通过 `MAX_HISTORY_TURNS` 注入最近历史，支持指代消解

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
