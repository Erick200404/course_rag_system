import os
import re
import hashlib
from pathlib import Path
from typing import List

from config import RERANK_TOP_K
import streamlit as st

from modules.pdf_loader import load_pdf
from modules.text_splitter import split_text
from modules.data_cleaner import filter_pages
from modules.embeddings import embed_chunks
from modules.vector_store import (
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    faiss_index_exists,
)
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
    # 如果目录不存在，则自动创建
    os.makedirs(save_dir, exist_ok=True)

    # 拼接最终保存路径，例如 data/test.pdf
    file_path = os.path.join(save_dir, uploaded_file.name)

    # 如果文件不存在，再写入，避免重复保存导致权限问题
    if not os.path.exists(file_path):
        # 以二进制形式写入文件内容
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    return file_path


def get_multi_pdf_storage_paths(pdf_paths: List[str]) -> tuple[str, str]:
    """
    根据多个 PDF 文件路径生成联合索引存储路径。

    逻辑：
    1. 取所有 PDF 文件名（不带后缀）
    2. 排序后拼接，避免上传顺序不同导致重复建库
    3. 使用 md5 生成固定长度的知识库 id
    4. 用该 id 作为 storage 子目录名称

    参数:
        pdf_paths: 多个 PDF 文件路径

    返回:
        index_path: FAISS 索引文件路径
        metadata_path: metadata 文件路径
    """
    # 取所有文件名（不带后缀）
    pdf_names = [Path(path).stem for path in pdf_paths]

    # 排序，避免同一组文件因顺序不同而得到不同索引目录
    pdf_names = sorted(pdf_names)

    # 先拼成一个字符串
    joined_name = "_".join(pdf_names)

    # 过滤中文和特殊字符，避免日志或路径显示混乱
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", joined_name)

    # 如果拼接后太长，使用 md5 保证路径稳定且不会过长
    kb_id = hashlib.md5(safe_name.encode("utf-8")).hexdigest()

    # 联合索引目录
    storage_dir = os.path.join("storage", kb_id)

    # FAISS 索引文件路径
    index_path = os.path.join(storage_dir, "faiss_index.bin")

    # metadata 文件路径
    metadata_path = os.path.join(storage_dir, "metadata.pkl")

    return index_path, metadata_path


def build_rag_pipeline(pdf_paths: List[str]):
    """
    根据多个 PDF 构建或加载完整的 RAG 检索基础。

    逻辑：
    1. 如果本地已有联合索引，则直接加载
    2. 否则逐个读取 PDF、过滤噪声页、切分 chunk
    3. 合并所有 PDF 的 chunk
    4. 统一生成 embedding、构建 FAISS 索引并保存

    参数:
        pdf_paths: 多个 PDF 文件路径

    返回:
        index: FAISS 索引
        metadata: 与向量一一对应的元数据列表
    """
    # 先根据当前这批 PDF 获取联合索引保存路径
    index_path, metadata_path = get_multi_pdf_storage_paths(pdf_paths)

    # 如果索引文件和 metadata 文件都已经存在，则直接加载
    if faiss_index_exists(index_path, metadata_path):
        index, metadata = load_faiss_index(index_path, metadata_path)
        return index, metadata

    # 如果本地没有索引，则从头开始构建
    all_chunks = []

    # 逐个处理每个 PDF
    for pdf_path in pdf_paths:
        # 1. 读取 PDF 内容（按页）
        pages = load_pdf(pdf_path)

        # 2. 过滤目录页、过短页等噪声页
        pages = filter_pages(pages)

        # 3. 将清洗后的页面切分为多个 chunk
        chunks = split_text(pages)

        # 4. 把当前 PDF 的 chunk 合并到总列表
        all_chunks.extend(chunks)

    # 5. 为所有 chunk 统一生成 embedding
    embedded_chunks = embed_chunks(all_chunks)

    # 6. 根据 embedding 构建 FAISS 索引，并抽取 metadata
    index, metadata = build_faiss_index(embedded_chunks)

    # 7. 将联合索引和 metadata 保存到本地，供下次直接复用
    save_faiss_index(index, metadata, index_path, metadata_path)

    return index, metadata


# 上传 PDF 的组件
uploaded_files = st.file_uploader(
    "请上传一个或多个 PDF 文件",
    type=["pdf"],
    accept_multiple_files=True
)

# 输入问题
question = st.text_input(
    "请输入你的问题：",
    placeholder="例如：请大致告诉我这个课件讲了什么？"
)

# 提问按钮
ask_button = st.button("开始提问")


# 如果用户已经上传文件，则给出提示
if uploaded_files:
    st.success(f"已上传 {len(uploaded_files)} 个文件")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")


# 当用户点击“开始提问”时，进入主流程
if ask_button:
    # 如果还没上传 PDF，则提示用户先上传
    if not uploaded_files:
        st.warning("请先上传 PDF 文件。")

    # 如果问题为空，则提示用户输入问题
    elif not question.strip():
        st.warning("请输入你的问题。")

    else:
        # 显示加载中的提示
        with st.spinner("正在加载索引并生成答案，请稍候..."):
            try:
                # 1. 先把上传的多个 PDF 保存到本地
                pdf_paths = [save_uploaded_file(uploaded_file) for uploaded_file in uploaded_files]

                # 2. 构建或加载多 PDF 联合索引
                #    如果已经存在索引，则直接加载
                #    否则自动完成 PDF 解析、清洗、切分、embedding、建索引、保存
                index, metadata = build_rag_pipeline(pdf_paths)

                # 3. 调用 RAG 问答链生成答案
                #    top_k=RERANK_TOP_K 表示取最相关的 RERANK_TOP_K 个 chunk 作为上下文
                result = generate_answer(question, index, metadata, top_k=RERANK_TOP_K)

                # 4. 显示最终答案
                st.subheader("回答结果")
                st.write(result["answer"])

                # 5. 显示召回到的片段，便于调试和观察检索效果
                st.subheader("召回片段预览")
                for i, item in enumerate(result["retrieved_chunks"], start=1):
                    with st.expander(
                        f"片段 {i} | 来源：{item['source']} | 第 {item['page']} 页"
                    ):
                        # 兼容不同检索阶段产生的分数显示
                        if "rerank_score" in item:
                            st.write(f"**Reranker 分数**：{item['rerank_score']:.4f}")
                        elif "score" in item:
                            st.write(f"**分数（L2 距离）**：{item['score']:.4f}")
                        elif "bm25_score" in item:
                            st.write(f"**BM25 分数**：{item['bm25_score']:.4f}")

                        # 显示召回到的 chunk 文本
                        st.write(item["text"])

            except Exception as e:
                # 如果运行出错，则在页面上显示错误信息
                st.error(f"运行出错：{e}")