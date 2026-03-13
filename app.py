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

# -----------------------------
# 自定义样式：让页面更像一个完整产品
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        padding-top: 1.2rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 100%);
        border: 1px solid #dbe7ff;
        margin-bottom: 1rem;
    }

    .section-card {
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 14px;
        border: 1px solid #e9eef5;
        background: #ffffff;
        margin-bottom: 1rem;
    }

    .answer-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border-left: 6px solid #4f8bf9;
        background: #f7fbff;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.92rem;
    }

    .file-tag {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        border-radius: 999px;
        background: #f3f6fb;
        border: 1px solid #dbe4f0;
        font-size: 0.9rem;
    }

    .metric-box {
        padding: 0.8rem 1rem;
        border-radius: 12px;
        background: #fafbfc;
        border: 1px solid #eef2f7;
        text-align: center;
    }

    .history-question {
        padding: 0.8rem 1rem;
        border-radius: 12px;
        background: #f6f7fb;
        border: 1px solid #e6eaf2;
        margin-bottom: 0.5rem;
    }

    .history-answer {
        padding: 0.8rem 1rem;
        border-radius: 12px;
        background: #f8fbff;
        border: 1px solid #dbe7ff;
        margin-bottom: 1rem;
    }

    .stButton > button {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 页面标题
st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom: 0.4rem;">📘 课程 PDF 智能问答系统</h1>
        <div class="small-muted">
            上传一个或多个课程 PDF，系统会基于 RAG 检索相关内容并生成回答。
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Session State 初始化
# 用于保存当前会话中的历史问答和输入框内容
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


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


def render_score(item: dict):
    """
    统一显示不同检索阶段可能出现的分数字段。
    """
    # 兼容不同检索阶段产生的分数显示
    if "rerank_score" in item:
        st.write(f"**Reranker 分数**：{item['rerank_score']:.4f}")
    elif "score" in item:
        st.write(f"**分数（L2 距离）**：{item['score']:.4f}")
    elif "bm25_score" in item:
        st.write(f"**BM25 分数**：{item['bm25_score']:.4f}")


def set_example_question(text: str):
    """
    设置示例问题到输入框。
    """
    st.session_state.question_input = text


# -----------------------------
# 侧边栏：系统配置区
# 这里不改后端接口，只做前端层面的可视配置
# -----------------------------
with st.sidebar:
    st.header("⚙️ 参数配置")

    # 是否显示召回片段，便于用户按需查看证据
    show_chunks = st.checkbox("显示召回片段", value=True)

    # 是否显示会话历史
    show_history = st.checkbox("显示历史问答", value=True)

    # 这里允许用户临时指定展示用的 top_k
    # 注意：最终传给 generate_answer 的 top_k 仍然是这里的值
    top_k = st.slider(
        "召回片段数量（Top-K）",
        min_value=1,
        max_value=10,
        value=min(RERANK_TOP_K, 10)
    )

    st.markdown("---")
    st.caption("提示：这里的 Top-K 会覆盖 config.py 中的默认展示值。")

    # 提供一个清空历史按钮，方便用户开始新一轮演示
    if st.button("🗑️ 清空历史问答", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.latest_result = None
        st.success("历史问答已清空。")


# -----------------------------
# 主区域布局：左侧知识库区，右侧问答区
# -----------------------------
left_col, right_col = st.columns([1, 2], gap="large")


with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📂 知识库管理")

    # 上传 PDF 的组件
    uploaded_files = st.file_uploader(
        "请上传一个或多个 PDF 文件",
        type=["pdf"],
        accept_multiple_files=True
    )

    # 如果用户已经上传文件，则给出提示
    if uploaded_files:
        st.success(f"已上传 {len(uploaded_files)} 个文件")

        # 这里将已上传文件显示成更清晰的标签形式
        for uploaded_file in uploaded_files:
            file_size_kb = len(uploaded_file.getvalue()) / 1024
            st.markdown(
                f"<div class='file-tag'>📄 {uploaded_file.name}（{file_size_kb:.1f} KB）</div>",
                unsafe_allow_html=True
            )

        # 如果已经上传文件，则提前计算索引路径，并显示当前知识库状态
        pdf_paths_preview = [os.path.join("data", uploaded_file.name) for uploaded_file in uploaded_files]
        index_path_preview, metadata_path_preview = get_multi_pdf_storage_paths(pdf_paths_preview)

        # 判断当前这组文件是否已有本地索引
        if faiss_index_exists(index_path_preview, metadata_path_preview):
            st.info("当前这组 PDF 已存在本地索引，可直接复用。")
        else:
            st.warning("当前这组 PDF 还没有本地索引，首次提问时会自动建库。")
    else:
        st.info("请先上传至少一个 PDF 文件。")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📌 快速提问示例")

    # 用多列放几个示例问题按钮，方便快速体验
    example_col1, example_col2 = st.columns(2)

    with example_col1:
        if st.button("总结主要内容", use_container_width=True):
            set_example_question("请总结这些课件的主要内容。")

        if st.button("核心知识点", use_container_width=True):
            set_example_question("请提炼这些文档中的核心知识点。")

    with example_col2:
        if st.button("涉及哪些模型", use_container_width=True):
            set_example_question("文档中一共提到了哪些模型？请分别说明。")

        if st.button("按页定位概念", use_container_width=True):
            set_example_question("这个概念主要出现在第几页？请给出对应依据。")

    st.markdown('</div>', unsafe_allow_html=True)

    # 这里给出一些可视化统计，让左侧信息更完整
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 当前状态")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("已上传文件数", len(uploaded_files) if uploaded_files else 0)

    with metric_col2:
        st.metric("当前 Top-K", top_k)

    st.markdown('</div>', unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💬 智能问答")

    # 使用 form 让输入和提交更像一个完整表单
    with st.form("qa_form", clear_on_submit=False):
        # 输入问题
        question = st.text_input(
            "请输入你的问题：",
            value=st.session_state.question_input,
            placeholder="例如：请大致告诉我这个课件讲了什么？"
        )

        # 提问按钮
        ask_button = st.form_submit_button("开始提问", use_container_width=True)

    # 每次 form 提交后，把输入框内容同步回 session_state
    st.session_state.question_input = question

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # 当用户点击“开始提问”时，进入主流程
    # -----------------------------
    if ask_button:
        # 如果还没上传 PDF，则提示用户先上传
        if not uploaded_files:
            st.warning("请先上传 PDF 文件。")

        # 如果问题为空，则提示用户输入问题
        elif not question.strip():
            st.warning("请输入你的问题。")

        else:
            try:
                # 用 status 代替单一 spinner，让用户更清楚当前步骤
                with st.status("正在处理你的问题...", expanded=True) as status:
                    # 第 1 步：先把上传的多个 PDF 保存到本地
                    st.write("1/4 正在保存上传的 PDF 文件...")
                    pdf_paths = [save_uploaded_file(uploaded_file) for uploaded_file in uploaded_files]

                    # 第 2 步：构建或加载多 PDF 联合索引
                    # 如果已经存在索引，则直接加载
                    # 否则自动完成 PDF 解析、清洗、切分、embedding、建索引、保存
                    st.write("2/4 正在构建或加载知识库索引...")
                    index, metadata = build_rag_pipeline(pdf_paths)

                    # 第 3 步：调用 RAG 问答链生成答案
                    # top_k 表示取最相关的 top_k 个 chunk 作为上下文
                    st.write("3/4 正在检索相关片段并生成回答...")
                    result = generate_answer(question, index, metadata, top_k=top_k)

                    # 第 4 步：保存结果到会话状态，便于展示历史
                    st.write("4/4 正在整理结果并更新界面...")
                    st.session_state.latest_result = result

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": result["answer"],
                            "retrieved_chunks": result["retrieved_chunks"],
                        }
                    )

                    status.update(label="回答生成完成", state="complete", expanded=False)

            except Exception as e:
                # 如果运行出错，则在页面上显示错误信息
                st.error(f"运行出错：{e}")

    # -----------------------------
    # 显示最新一次回答结果
    # -----------------------------
    if st.session_state.latest_result is not None:
        result = st.session_state.latest_result

        # 4. 显示最终答案
        st.subheader("✅ 回答结果")
        st.markdown(
            f"""
            <div class="answer-card">
                {result["answer"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        # 5. 显示召回到的片段，便于调试和观察检索效果
        if show_chunks:
            st.subheader("📎 召回片段预览")
            for i, item in enumerate(result["retrieved_chunks"], start=1):
                # 为了兼容字段缺失，这里给 source/page 设置默认值
                source = item.get("source", "未知来源")
                page = item.get("page", "未知页码")

                with st.expander(
                    f"片段 {i} | 来源：{source} | 第 {page} 页"
                ):
                    # 显示分数信息
                    render_score(item)

                    # 显示召回到的 chunk 文本
                    st.write(item.get("text", ""))

    # -----------------------------
    # 显示历史问答
    # -----------------------------
    if show_history and st.session_state.chat_history:
        st.subheader("🕘 历史问答")

        # 倒序显示，让最新的问题排在最前面
        for i, record in enumerate(reversed(st.session_state.chat_history), start=1):
            st.markdown(
                f"""
                <div class="history-question">
                    <strong>用户问题 {i}</strong><br>
                    {record["question"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="history-answer">
                    <strong>系统回答</strong><br>
                    {record["answer"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            # 历史问答中的证据片段也可以展开查看
            if show_chunks:
                with st.expander(f"查看这轮问答的召回依据（共 {len(record['retrieved_chunks'])} 条）"):
                    for j, item in enumerate(record["retrieved_chunks"], start=1):
                        source = item.get("source", "未知来源")
                        page = item.get("page", "未知页码")

                        st.markdown(f"**片段 {j} | 来源：{source} | 第 {page} 页**")
                        render_score(item)
                        st.write(item.get("text", ""))
                        st.markdown("---")