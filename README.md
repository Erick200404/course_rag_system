**1. 项目名称**
 基于 RAG 的课程 PDF 智能问答与复习辅助系统

**2. 项目背景**
 大学生在期末复习时需要频繁翻阅老师提供的 PDF 课件和讲义，查找知识点效率低，难以快速定位答案出处。

**3. 项目目标**
 实现一个支持多 PDF 上传、内容检索问答、引用页码展示和复习辅助的智能系统。

**4. 第一版功能**
 PDF 解析、文本切分、向量检索、问答生成、引用溯源。

## 快速启动
请在anconda prompt中输入  
E:  
cd E:\pycharm_workspace\course_rag_system  
conda activate course_rag  

## Roadmap
v1.0
- [x] PDF loader
- [x] Text splitter
- [x] Embedding module
- [x] Vector store (FAISS)
- [x] Retriever
- [x] RAG pipeline
- [x] Streamlit UI
-----------------------------
v2.0
- [x] 修改Vector store，实现数据持久化
- [ ] Hybrid retrieval (BM25 + Vector)
- [ ] Reranker
### pdf_loader.py
目前只能做文本pdf的识别，待完善

### text_split.py
目前的逻辑是按字符硬切，对目录页不友好，可能把语义切断，待完善

### embeddings.py
目前是直接用的text-embedding-3-small，后续考虑使用本地部署的模型

### vector_store.py
v1.0 把这些向量放进 FAISS 里，后面用户提问时就能查最相近的 chunk  
v2.0 考虑把向量保存到数据库里，方便后续查询

### retriever.py
根据用户问题，在 FAISS 中检索最相近的 top-k 个 chunk

### rag_chain.py
把召回的 chunk 拼成上下文，发给聊天模型，让它回答。即实现RAG问答链路，结合检索结果调用大模型生成答案

### run.py
使用Streamlit UI构建前端  
接入索引复用功能