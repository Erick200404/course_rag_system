import os
from pathlib import Path

import streamlit as st

from modules.pdf_loader import load_pdf
from modules.text_splitter import split_text
from modules.embeddings import embed_chunks
from modules.vector_store import build_faiss_index
from modules.rag_chain import generate_answer


# 设置页面基础信息
st.set_page_config(
    page_title="课程 PDF 智能问答系统",
    page_icon="📘",
    layout="wide"
)

# 页面标题
st.title("📘 课程 PDF 智能问答系统")
st.write("上传课程 PDF 后，输入问题，系统会基于 RAG 检索相关内容并生成回答。")


def save_uploaded_file(uploaded_file, save_dir: str = "data") -> str:
    """
    保存用户上传的 PDF 文件到本地 data 目录。

    参数:
        uploaded_file: Streamlit 上传的文件对象
        save_dir: 保存目录

    返回:
        保存后的文件路径
    """
    # 如果 data 目录不存在，就创建
    os.makedirs(save_dir, exist_ok=True)

    # 拼接保存路径
    file_path = os.path.join(save_dir, uploaded_file.name)

    # 以二进制方式写入文件
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def build_rag_pipeline(pdf_path: str):
    """
    根据 PDF 构建完整的 RAG 检索基础：
    1. 读取 PDF
    2. 切分 chunk
    3. 生成 embedding
    4. 构建 FAISS 索引

    返回:
        index: FAISS 索引
        metadata: 元数据列表
    """
    # 1. 读取 PDF
    pages = load_pdf(pdf_path)

    # 2. 切分文本
    chunks = split_text(pages)

    # 3. 生成文本向量
    embedded_chunks = embed_chunks(chunks)

    # 4. 构建 FAISS 索引
    index, metadata = build_faiss_index(embedded_chunks)

    return index, metadata


# 上传 PDF 的组件
uploaded_file = st.file_uploader("请上传一个 PDF 文件", type=["pdf"])

# 输入问题
question = st.text_input("请输入你的问题：", placeholder="例如：BETL 的总体框架是什么？")

# 提问按钮
ask_button = st.button("开始提问")


# 如果用户上传了文件
if uploaded_file is not None:
    st.success(f"已上传文件：{uploaded_file.name}")

# 如果点击了提问按钮
if ask_button:
    # 先检查是否上传文件
    if uploaded_file is None:
        st.warning("请先上传 PDF 文件。")

    # 再检查是否输入问题
    elif not question.strip():
        st.warning("请输入你的问题。")

    else:
        # 用 spinner 提示用户当前正在处理
        with st.spinner("正在解析 PDF、构建索引并生成答案，请稍候..."):
            try:
                # 1. 保存上传的 PDF
                pdf_path = save_uploaded_file(uploaded_file)

                # 2. 构建 RAG 基础索引
                index, metadata = build_rag_pipeline(pdf_path)

                # 3. 生成答案
                result = generate_answer(question, index, metadata, top_k=3)

                # 4. 显示回答
                st.subheader("回答结果")
                st.write(result["answer"])

                # 5. 显示召回片段
                st.subheader("召回片段预览")
                for i, item in enumerate(result["retrieved_chunks"], start=1):
                    with st.expander(f"片段 {i} | 来源：{item['source']} | 第 {item['page']} 页"):
                        st.write(f"**分数（L2 距离）**：{item['score']:.4f}")
                        st.write(item["text"])

            except Exception as e:
                st.error(f"运行出错：{e}")