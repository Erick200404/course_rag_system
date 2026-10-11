import os
from pathlib import Path
from typing import List

import requests
import streamlit as st

from config import MAX_HISTORY_TURNS, RERANK_TOP_K, RAG_FLOW
from modules.document_loader import SUPPORTED_EXTENSIONS


# ==============================
# FastAPI 服务地址配置
# ==============================
# 这里使用环境变量优先，方便以后部署时切换地址；
# 如果没有配置环境变量，则默认请求本机的 FastAPI 服务。
API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")


# 设置页面基础信息
st.set_page_config(
    page_title="课程资料智能问答系统",
    page_icon="📘",
    layout="wide",
)


# -----------------------------
# 自定义样式：Perplexity 风格（羊皮纸底色 + 单一墨绿点缀 + 扁平紧凑）
# 仅借鉴其视觉语言，布局按本项目逻辑（左侧资料库 + 居中 hero 提问）重组
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --parchment: #faf8f5;
        --soft-paper: #fdfbfa;
        --rail: #f4f1ec;
        --warm-mist: #d1d1cd;
        --ash: #92918b;
        --graphite: #72706b;
        --ink: #27251e;
        --teal: #016a71;
        --shadow-subtle: rgba(0, 0, 0, 0.08) 0px 1px 2px;
    }

    /* 整体画布：所有全屏背景都用羊皮纸，不用纯白 */
    html, body, .stApp { background: var(--parchment); }
    .stApp { color: var(--ink); }

    /* 收起 Streamlit 默认顶部菜单与页脚，降低 chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { display: none; }

    /* 主内容列：居中、窄栏（贴近 Perplexity 的 900px 限制） */
    .block-container {
        max-width: 960px !important;
        margin: 0 auto;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    /* 左栏：略深的羊皮纸导轨，右侧发丝边 */
    section[data-testid="stSidebar"] {
        background: var(--rail);
        border-right: 1px solid var(--warm-mist);
    }
    section[data-testid="stSidebar"] .block-container {
        max-width: 100% !important;
        padding: 1.6rem 1.1rem;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: var(--ink); font-weight: 500; }

    /* 文本输入：hero 搜索框 */
    .stTextInput > div > div > input {
        background: var(--parchment);
        border: 1px solid var(--warm-mist);
        border-radius: 12px;
        color: var(--ink);
        padding: 14px 16px;
        font-size: 16px;
        line-height: 1.5;
    }
    .stTextInput > div > div > input::placeholder { color: var(--graphite); }
    .stTextInput > div > div > input:focus {
        border-color: var(--teal);
        box-shadow: 0 0 0 3px rgba(1, 106, 113, 0.15);
    }
    /* hero 输入框的静态墨绿微光（模板允许的唯一点缀效果） */
    .hero-input .stTextInput > div > div {
        box-shadow: 0 0 0 4px rgba(1, 106, 113, 0.08);
        border-radius: 12px;
    }

    /* 按钮：默认 ghost（石墨文字 + 发丝边 + 6px 圆角） */
    .stButton > button,
    .stFormSubmitButton > button {
        background: transparent;
        color: var(--graphite);
        border: 1px solid var(--warm-mist);
        border-radius: 6px;
        font-weight: 500;
        font-size: 14px;
        padding: 8px 14px;
        transition: color .15s, border-color .15s;
    }
    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        color: var(--ink);
        border-color: var(--graphite);
    }

    /* 主操作按钮：墨色填充（hero 提交） */
    .primary-btn button {
        background: var(--ink) !important;
        color: var(--parchment) !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        padding: 10px 22px !important;
    }
    .primary-btn button:hover { background: #1b1913 !important; color: var(--parchment) !important; }

    /* 示例问题 / 资料芯片：pill（9999px 圆角） */
    .pill-row button {
        border-radius: 9999px !important;
        border: 1px solid var(--warm-mist) !important;
        background: transparent !important;
        color: var(--ink) !important;
        font-size: 14px !important;
        text-align: left !important;
    }
    .pill-row button:hover { border-color: var(--teal) !important; color: var(--teal) !important; }

    /* 卡片：柔纸表面 + 16px 圆角 + 发丝边（扁平，不用重阴影） */
    .card {
        background: var(--soft-paper);
        border: 1px solid var(--warm-mist);
        border-radius: 16px;
        padding: 16px;
    }

    /* 答案卡片 */
    .answer-card {
        background: var(--soft-paper);
        border: 1px solid var(--warm-mist);
        border-radius: 16px;
        padding: 18px 20px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .answer-card p { margin: 0 0 0.6rem 0; }

    /* 历史问答气泡 */
    .history-question {
        padding: 10px 14px;
        border-radius: 12px;
        background: var(--parchment);
        border: 1px solid var(--warm-mist);
        margin-bottom: 0.4rem;
        color: var(--ink);
    }
    .history-answer {
        padding: 10px 14px;
        border-radius: 12px;
        background: var(--soft-paper);
        border: 1px solid var(--warm-mist);
        margin-bottom: 1rem;
        color: var(--ink);
        line-height: 1.6;
    }

    /* 区块小标题（替代 emoji 标题，保持克制） */
    .section-label {
        color: var(--graphite);
        font-size: 12px;
        font-weight: 500;
        letter-spacing: .04em;
        margin: 0 0 0.6rem 0;
    }

    /* 资料条目（侧栏「资料库」） */
    .doc-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--warm-mist);
        background: var(--parchment);
        margin-bottom: 6px;
        font-size: 13px;
        color: var(--ink);
    }
    .doc-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .doc-item .size { color: var(--ash); font-size: 11px; }

    /* 资料芯片（hero 下方：本次将基于哪些资料） */
    .doc-chip {
        display: inline-flex;
        align-items: center;
        border: 1px solid var(--warm-mist);
        border-radius: 9999px;
        padding: 4px 10px;
        margin: 3px 4px 3px 0;
        font-size: 12px;
        color: var(--ink);
        background: var(--soft-paper);
    }

    /* 状态点 */
    .status-chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--graphite); }
    .status-chip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--c); }

    /* 墨绿徽标（NEW / 编排模式） */
    .badge-teal {
        display: inline-block;
        background: var(--teal);
        color: #fff;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 500;
        padding: 2px 8px;
    }

    /* 文案层级 */
    .muted { color: var(--graphite); font-size: 13px; }
    .caption-teal { color: var(--teal); font-size: 12px; }
    .caption-warn { color: #b4543f; font-size: 12px; }

    /* 折叠面板：羊皮纸风格 */
    .streamlit-expanderHeader {
        background: var(--soft-paper) !important;
        border: 1px solid var(--warm-mist) !important;
        border-radius: 12px !important;
        color: var(--ink) !important;
        font-size: 13px !important;
    }

    /* 分隔线：发丝灰 */
    hr { border: none; border-top: 1px solid var(--warm-mist); margin: 1rem 0; }

    /* 隐藏 sidebar 内部多余留白导致的滚动条影响 */
    section[data-testid="stSidebar"] .stFileUploader { margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
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
    保存用户上传的资料文件到本地 data 目录。

    参数:
        uploaded_file: Streamlit 上传的文件对象
        save_dir: 保存目录

    返回:
        保存后的文件路径
    """
    # 如果目录不存在，则自动创建
    os.makedirs(save_dir, exist_ok=True)

    # Streamlit 正常只返回文件名；显式拒绝路径，避免客户端构造路径穿越。
    if Path(uploaded_file.name).name != uploaded_file.name:
        raise ValueError("上传文件名不能包含路径")

    # 拼接最终保存路径，例如 data/test.pdf
    file_path = os.path.join(save_dir, uploaded_file.name)

    # 原子替换保证同名文件内容更新后一定覆盖旧版本；
    # 否则 content hash 会继续基于旧文件计算，导致错误复用旧索引。
    temp_path = f"{file_path}.uploading"
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        os.replace(temp_path, file_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

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


def call_chat_api(
    question: str,
    doc_names: List[str],
    top_k: int,
    history: List[dict] | None = None,
) -> dict:
    """
    调用 FastAPI 的 /chat 接口，获取 RAG 问答结果。

    参数:
        question: 用户问题
        doc_names: 参与问答的文档文件名列表
        top_k: 最终召回片段数量
        history: 对话历史（CRAG 模式下用于多轮查询改写）

    返回:
        FastAPI 返回的 JSON 结果
    """
    # 组织请求体，和后端 ChatRequest 保持一致
    payload = {
        "question": question,
        "doc_names": doc_names,
        "top_k": top_k,
    }

    # 传入最近几轮对话，让后端能改写「它指什么」这类依赖上下文的问题
    if history:
        payload["history"] = history[-MAX_HISTORY_TURNS:]

    try:
        # 发送 POST 请求到 FastAPI 后端
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=120,
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
    if "rerank_score" in item:
        st.write(f"**Reranker 分数**：{item['rerank_score']:.4f}")
    elif "score" in item:
        st.write(f"**相似度分数**：{item['score']:.4f}")
    elif "bm25_score" in item:
        st.write(f"**BM25 分数**：{item['bm25_score']:.4f}")


def set_example_question(text: str):
    """设置示例问题到输入框。"""
    st.session_state.question_input = text


def render_status_chip(ok: bool):
    """用墨绿/克制的红点展示后端在线状态。"""
    color = "#016a71" if ok else "#b4543f"
    label = "后端在线" if ok else "后端离线"
    st.markdown(
        f'<span class="status-chip"><span class="dot" style="--c:{color}"></span>{label}</span>',
        unsafe_allow_html=True,
    )


# ============================================================
# 左侧栏：资料库 + 设置（突出展示「传进去的资料」）
# ============================================================
with st.sidebar:
    # 品牌锚点
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <span style="font-size:18px;">📘</span>
            <span style="font-size:16px;font-weight:500;color:var(--ink);">课程问答</span>
        </div>
        <div class="muted" style="margin-bottom:1rem;">RAG 智能问答 · 资料检索与复习</div>
        """,
        unsafe_allow_html=True,
    )

    # 编排模式徽标（体现当前 RAG_FLOW）
    st.markdown(
        f'<div style="margin-bottom:1rem;"><span class="badge-teal">编排模式：{RAG_FLOW.upper()}</span></div>',
        unsafe_allow_html=True,
    )

    # —— 资料库 ——
    st.markdown('<div class="section-label">我的资料库</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "上传课程资料",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help="支持 PDF / Word / PPT / Markdown / TXT，可一次上传多个",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            size_kb = len(uploaded_file.getvalue()) / 1024
            name = uploaded_file.name
            st.markdown(
                f'<div class="doc-item"><span>📄</span>'
                f'<span class="name" title="{name}">{name}</span>'
                f'<span class="size">{size_kb:.0f}KB</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="muted" style="margin-top:.4rem;">已载入 {len(uploaded_files)} 份资料，将作为本次问答的知识来源。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="muted">尚未上传资料。上传后系统会自动建库，即可开始提问。</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # —— 设置 ——
    st.markdown('<div class="section-label">设置</div>', unsafe_allow_html=True)

    show_chunks = st.checkbox("显示召回片段", value=True)
    show_history = st.checkbox("显示历史问答", value=True)

    top_k = st.slider(
        "召回片段数量（Top-K）",
        min_value=1,
        max_value=10,
        value=min(RERANK_TOP_K, 10),
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # 后端在线状态
    backend_ok = call_health_api()
    render_status_chip(backend_ok)
    if not backend_ok:
        st.markdown(
            '<div class="caption-warn" style="margin-top:.4rem;">请先启动后端：uvicorn api.main:app --reload</div>',
            unsafe_allow_html=True,
        )

    if st.button("清空历史问答", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.latest_result = None
        st.rerun()


# ============================================================
# 主区域：居中 hero 提问 + 示例 + 结果
# ============================================================
# 顶部 wordmark（克制，不用大字重）
st.markdown(
    """
    <div style="text-align:center;margin-bottom:1.4rem;">
        <div style="font-size:22px;font-weight:500;color:var(--ink);">课程资料智能问答</div>
        <div class="muted" style="margin-top:.3rem;">上传资料，用自然语言提问，基于检索内容生成带引用的回答。</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# —— Hero 提问框 ——
st.markdown('<div class="hero-input">', unsafe_allow_html=True)
with st.form("qa_form", clear_on_submit=False):
    question = st.text_input(
        "question_input",
        value=st.session_state.question_input,
        placeholder="例如：请大致告诉我这个课件讲了什么？",
        label_visibility="collapsed",
    )
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    ask_button = st.form_submit_button("提问", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# 同步输入框内容
st.session_state.question_input = question

# hero 下方：本次将基于的资料芯片（直观体现「传进去的资料」）
if uploaded_files:
    chips = "".join(
        f'<span class="doc-chip">📄 {uf.name}</span>' for uf in uploaded_files
    )
    st.markdown(
        f'<div class="muted" style="margin:.6rem 0 .2rem;">本次问答将基于以下资料：</div>{chips}',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="muted" style="margin:.6rem 0;">尚未上传资料，先到左侧「我的资料库」上传课件吧。</div>',
        unsafe_allow_html=True,
    )

# —— 示例问题：pill 芯片（2 列）——
st.markdown('<div class="section-label" style="margin-top:1.6rem;">试试这样问</div>', unsafe_allow_html=True)
st.markdown('<div class="pill-row">', unsafe_allow_html=True)
ex_col1, ex_col2 = st.columns(2)
with ex_col1:
    if st.button("总结这些课件的主要内容", use_container_width=True, key="ex1"):
        set_example_question("请总结这些课件的主要内容。")
    if st.button("提炼核心知识点", use_container_width=True, key="ex2"):
        set_example_question("请提炼这些文档中的核心知识点。")
with ex_col2:
    if st.button("文档中提到了哪些模型", use_container_width=True, key="ex3"):
        set_example_question("文档中一共提到了哪些模型？请分别说明。")
    if st.button("某个概念出现在第几页", use_container_width=True, key="ex4"):
        set_example_question("这个概念主要出现在第几页？请给出对应依据。")
st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# 主流程：提交后调用后端并展示结果
# -----------------------------
if ask_button:
    if not uploaded_files:
        st.warning("请先上传课程资料文件。")
    elif not question.strip():
        st.warning("请输入你的问题。")
    elif not backend_ok:
        st.error("FastAPI 后端不可用，请先运行：uvicorn api.main:app --reload")
    else:
        try:
            with st.status("正在处理你的问题...", expanded=True) as status:
                status.write("1/3 正在保存上传的文件...")
                doc_paths = [save_uploaded_file(uploaded_file) for uploaded_file in uploaded_files]
                doc_names = [os.path.basename(path) for path in doc_paths]

                status.write("2/3 正在调用 FastAPI 问答接口...")
                result = call_chat_api(
                    question=question.strip(),
                    doc_names=doc_names,
                    top_k=top_k,
                    history=st.session_state.chat_history,
                )

                status.write("3/3 正在更新界面...")
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
            st.error(f"运行出错：{e}")


# -----------------------------
# 展示最新一次回答结果
# -----------------------------
if st.session_state.latest_result is not None:
    result = st.session_state.latest_result

    st.markdown('<div class="section-label" style="margin-top:1.8rem;">回答</div>', unsafe_allow_html=True)
    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.markdown(result["answer"])
    st.markdown("</div>", unsafe_allow_html=True)

    # CRAG 流程信息：改写次数与是否走兜底
    if result.get("flow") == "crag":
        if result.get("fallback"):
            st.markdown(
                '<div class="caption-warn">⚠ 检索结果不足以回答，已尝试改写查询后仍走兜底回答。</div>',
                unsafe_allow_html=True,
            )
        elif result.get("retries"):
            st.markdown(
                f'<div class="caption-teal">🔁 查询经过 {result["retries"]} 次改写重试后命中相关资料。</div>',
                unsafe_allow_html=True,
            )
        if result.get("question") and result["question"] != result.get(
            "original_question", result["question"]
        ):
            st.markdown(
                f'<div class="caption-teal">改写后的查询：{result["question"]}</div>',
                unsafe_allow_html=True,
            )

    # 召回片段预览
    if show_chunks:
        st.markdown('<div class="section-label" style="margin-top:1.2rem;">召回片段预览</div>', unsafe_allow_html=True)
        for i, item in enumerate(result["retrieved_chunks"], start=1):
            source = item.get("source", "未知来源")
            page = item.get("page", "未知页码")
            heading = item.get("heading", "")

            label = f"片段 {i} · 来源：{source} · 第 {page} 页"
            if heading:
                label += f" · {heading}"

            image_path = item.get("image_path", "")
            if image_path:
                label += " · 课件图片"

            with st.expander(label):
                render_score(item)
                st.write(item.get("text", ""))
                if image_path and os.path.exists(image_path):
                    st.image(image_path)


# -----------------------------
# 历史问答
# -----------------------------
if show_history and st.session_state.chat_history:
    st.markdown('<div class="section-label" style="margin-top:1.8rem;">历史问答</div>', unsafe_allow_html=True)

    for i, record in enumerate(reversed(st.session_state.chat_history), start=1):
        st.markdown(
            f"""
            <div class="history-question"><strong>问题 {i}</strong><br>{record["question"]}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="history-answer">{record["answer"]}</div>
            """,
            unsafe_allow_html=True,
        )

        if show_chunks:
            with st.expander(f"查看这轮问答的召回依据（共 {len(record['retrieved_chunks'])} 条）"):
                for j, item in enumerate(record["retrieved_chunks"], start=1):
                    source = item.get("source", "未知来源")
                    page = item.get("page", "未知页码")
                    st.markdown(f"**片段 {j} · 来源：{source} · 第 {page} 页**")
                    render_score(item)
                    st.write(item.get("text", ""))
                    st.markdown("---")
