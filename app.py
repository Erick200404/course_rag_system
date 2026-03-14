import os
from typing import List

import requests
import streamlit as st

from config import RERANK_TOP_K


# ==============================
# FastAPI 服务地址配置
# ==============================
# 这里使用环境变量优先，方便以后部署时切换地址；
# 如果没有配置环境变量，则默认请求本机的 FastAPI 服务。
API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")


# 设置页面基础信息
st.set_page_config(
    page_title="课程 PDF 智能问答系统",
    page_icon="📘",
    layout="wide"
)

# -----------------------------
# 自定义样式：尽量保持你原来的页面风格
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


def call_health_api() -> bool:
    """
    调用 FastAPI 的健康检查接口，用于判断后端服务是否在线。

    返回:
        True  -> 后端服务正常
        False -> 后端服务不可用
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def call_chat_api(question: str, pdf_names: List[str], top_k: int) -> dict:
    """
    调用 FastAPI 的 /chat 接口，获取 RAG 问答结果。

    参数:
        question: 用户问题
        pdf_names: 参与问答的 PDF 文件名列表
        top_k: 最终召回片段数量

    返回:
        FastAPI 返回的 JSON 结果
    """
    # 组织请求体，和后端 ChatRequest 保持一致
    payload = {
        "question": question,
        "pdf_names": pdf_names,
        "top_k": top_k
    }

    try:
        # 发送 POST 请求到 FastAPI 后端
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=120
        )

        # 如果后端返回非 200，抛出异常并带上返回内容
        response.raise_for_status()

        # 解析并返回 JSON 结果
        return response.json()

    except requests.HTTPError as e:
        # 优先展示后端返回的 detail，方便定位问题
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text

        raise RuntimeError(f"后端接口调用失败：{error_detail}") from e

    except requests.RequestException as e:
        # 处理连接失败、超时等请求层异常
        raise RuntimeError(f"无法连接 FastAPI 服务：{e}") from e


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
# -----------------------------
with st.sidebar:
    st.header("⚙️ 参数配置")

    # 是否显示召回片段，便于用户按需查看证据
    show_chunks = st.checkbox("显示召回片段", value=True)

    # 是否显示会话历史
    show_history = st.checkbox("显示历史问答", value=True)

    # 允许用户临时指定展示用的 top_k
    top_k = st.slider(
        "召回片段数量（Top-K）",
        min_value=1,
        max_value=10,
        value=min(RERANK_TOP_K, 10)
    )

    st.markdown("---")
    st.caption("提示：这里的 Top-K 会传给 FastAPI 后端接口。")

    # 显示后端服务状态，方便调试
    backend_ok = call_health_api()
    if backend_ok:
        st.success("FastAPI 后端服务正常")
    else:
        st.error("FastAPI 后端不可用，请先启动 api/main.py")

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

        # 将已上传文件显示成更清晰的标签形式
        for uploaded_file in uploaded_files:
            file_size_kb = len(uploaded_file.getvalue()) / 1024
            st.markdown(
                f"<div class='file-tag'>📄 {uploaded_file.name}（{file_size_kb:.1f} KB）</div>",
                unsafe_allow_html=True
            )

        # 提示当前系统改为通过 FastAPI 处理问答
        st.info("提问时会将文件保存到本地 data 目录，并由 FastAPI 后端完成建库与问答。")
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

    # 这里给出一些可视化统计
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

        # 如果后端服务不可用，则提示先启动 FastAPI
        elif not backend_ok:
            st.error("FastAPI 后端不可用，请先运行：uvicorn api.main:app --reload")

        else:
            try:
                # 用 status 展示当前处理进度
                with st.status("正在处理你的问题...", expanded=True) as status:
                    # 第 1 步：将上传的 PDF 保存到本地 data 目录
                    st.write("1/3 正在保存上传的 PDF 文件...")
                    pdf_paths = [save_uploaded_file(uploaded_file) for uploaded_file in uploaded_files]

                    # 从路径中提取文件名，传给后端接口
                    pdf_names = [os.path.basename(path) for path in pdf_paths]

                    # 第 2 步：调用 FastAPI 后端接口
                    st.write("2/3 正在调用 FastAPI 问答接口...")
                    result = call_chat_api(
                        question=question.strip(),
                        pdf_names=pdf_names,
                        top_k=top_k
                    )

                    # 第 3 步：保存结果到会话状态，便于展示历史
                    st.write("3/3 正在更新界面...")
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

        st.subheader("✅ 回答结果")
        st.markdown(
            f"""
            <div class="answer-card">
                {result["answer"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        # 显示召回到的片段，便于调试和观察检索效果
        if show_chunks:
            st.subheader("📎 召回片段预览")
            for i, item in enumerate(result["retrieved_chunks"], start=1):
                source = item.get("source", "未知来源")
                page = item.get("page", "未知页码")

                with st.expander(
                    f"片段 {i} | 来源：{source} | 第 {page} 页"
                ):
                    render_score(item)
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