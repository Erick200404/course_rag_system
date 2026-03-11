**1. 项目名称**
 基于 RAG 的课程 PDF 智能问答与复习辅助系统

**2. 项目背景**
 大学生在期末复习时需要频繁翻阅老师提供的 PDF 课件和讲义，查找知识点效率低，难以快速定位答案出处。

**3. 项目目标**
 实现一个支持多 PDF 上传、内容检索问答、引用页码展示和复习辅助的智能系统。

**4. 第一版功能**
 PDF 解析、文本切分、向量检索、问答生成、引用溯源。

## Roadmap

- [x] PDF loader
- [x] Text splitter
- [x] Embedding module
- [ ] Vector store (FAISS)
- [ ] Retriever
- [ ] RAG pipeline
- [ ] Streamlit UI
- [ ] Hybrid retrieval (BM25 + Vector)
- [ ] Reranker
### pdf_loader.py
目前只能做文本pdf的识别，待完善

### text_split.py
目前的逻辑是按字符硬切，对目录页不友好，可能把语义切断，待完善

### embeddings.py
目前是直接用的text-embedding-3-small，后续考虑使用本地部署的模型