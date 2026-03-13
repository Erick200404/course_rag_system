# 基于 RAG 的课程 PDF 智能问答与复习辅助系统

## 一、项目简介

本项目实现了一个 **基于 Retrieval-Augmented Generation (RAG)** 的课程资料智能问答系统。

系统允许用户上传课程 PDF（课件、讲义、实验报告等），并通过自然语言提问，系统会自动检索相关内容并生成答案，同时给出 **引用来源（文件名 + 页码）**，帮助用户快速定位知识点。

该项目主要用于探索 **大语言模型 + 信息检索技术** 在学习辅助场景中的应用。
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

整体 RAG Pipeline 如下：

```
PDF
 ↓
PDF Loader
 ↓
Text Cleaner
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
FAISS
BM25
BGE Reranker
OpenAI Compatible API
Streamlit
PyMuPDF
```

核心技术包括：

- Retrieval-Augmented Generation (RAG)
- Hybrid Retrieval（向量检索 + 关键词检索）
- Reranking
- 多文档知识库

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
在项目根目录创建 `data`和`storage`目录：  
在项目根目录创建 `.env` 文件并按照如下示例填写：

```
ZHI_API_KEY=你的APIKEY
ZHI_BASE_URL=你的API接口地址
```
------

## 3 启动系统

进入项目目录：

```
cd E:\pycharm_workspace\course_rag_system
```

启动 Streamlit：

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

# v1.0 基础 RAG 系统

实现最基础的 RAG pipeline：

-  [x] PDF Loader
-  [x] Text Splitter
-  [x] Embedding 模块
-  [x] Vector Store（FAISS）
-  [x] Retriever
-  [x] RAG Pipeline
-  [x] Streamlit UI

系统可以：

- 上传 PDF
- 提问
- 检索相关内容
- 生成回答

------

# v2.0 检索能力优化

提升检索质量。

新增：

-  [x] Vector Store 数据持久化
-  [x] Hybrid Retrieval（BM25 + Vector）
-  [x] Reranker 重排序

改进：

- 检索召回质量明显提升
- 回答相关性提高

------

# v3.0 工程结构优化

对系统结构进行优化：

-  [x] 优化 Text Splitter
-  [x] 引入统一 `config` 管理参数
-  [x] BM25 中文分词优化
-  [x] 检索参数可配置化

------

# v4.0 多文档知识库

支持多 PDF 共同组成知识库：

-  [x] 多 PDF 上传
-  [x] 联合向量索引
-  [x] 跨文档检索
-  [x] 引用来源展示（文件名 + 页码）

系统从：

```
单文档问答
```

升级为：

```
多文档知识库问答
```

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
- 使用 chunk_size + overlap

当前问题：

- 按字符切分
- 可能会打断语义

后续可优化：

- 语义分割
- 标题增强

------

## embeddings.py

负责：

- 调用 embedding API
- 生成文本向量

当前使用：

```
text-embedding-3-small
```

未来计划：

- 支持本地 embedding 模型

------

## vector_store.py

负责：

- 构建 FAISS 向量索引
- 保存 metadata
- 支持索引持久化

功能：

- 向量存储
- 相似度搜索

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

------

# 未来优化方向

未来可以继续优化：

-  语义级文本切分
-  Query Rewrite
-  Answer Evaluation
-  本地 embedding 模型
-  向量数据库（Milvus / Qdrant）
-  支持更多文档格式（docx、ppt）

------

# 项目总结

本项目实现了一个 **完整的 RAG 问答系统原型**，包括：

- 文档解析
- 知识检索
- 检索优化
- 多文档知识库
- 问答生成
- 引用溯源

适用于：

- 课程资料问答
- 学习辅助
- 小型知识库问答系统