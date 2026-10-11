"""Course libraries and persistent conversations, served by FastAPI."""

import hashlib
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from config import RERANK_TOP_K
from modules.api_client import api_request, upload_document
from modules.document_loader import SUPPORTED_EXTENSIONS
from modules.file_storage import save_uploaded_file  # compatibility for older callers

st.set_page_config(
    page_title="课程问答 · 我的学习桌", page_icon=":material/menu_book:",
    layout="wide", initial_sidebar_state="expanded",
)

# Keep the native header and sidebar toggle visible. Hiding stHeader made
# the old sidebar impossible to reopen once collapsed.
st.html("""
<style>
    .stMain .block-container { max-width: 900px; padding-top: 3.5rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-weight: 500 !important; }
    header[data-testid="stHeader"] { background: #faf8f5; }
    [data-testid="stSidebar"] { border-right: 1px solid #d1d1cd; }
    [data-testid="stChatMessage"] { border-radius: 16px; }
    [data-testid="stForm"] { background: #fdfbfa; border-radius: 16px; }
    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: #27251e; color: #faf8f5; border-color: #27251e;
    }
    [data-testid="stFileUploaderDropzone"] { background: #fdfbfa; }
</style>
""")

st.session_state.setdefault("upload_receipts", set())
reset = st.session_state.pop("pending_navigation_reset", None)
if reset:
    for key in (("course", "session") if reset == "course" else ("session",)):
        st.session_state.pop(key, None)
        st.query_params.pop(key, None)
deleted = st.session_state.pop("pending_deleted_document", None)
if deleted:
    deleted_course, deleted_name, affected_sessions = deleted
    st.session_state.upload_receipts = {
        receipt for receipt in st.session_state.upload_receipts
        if not (receipt[0] == deleted_course and receipt[1] == deleted_name)
    }
    for affected in affected_sessions:
        key = f"documents_{affected}"
        if key in st.session_state:
            st.session_state[key] = [name for name in st.session_state[key] if name != deleted_name]
# Native selectbox URL binding serializes format_func's display labels.
# Course names may repeat and session titles change after the first answer,
# so route this app with immutable IDs instead of mutable display labels.
for pending, target in (("pending_course", "course"), ("pending_session", "session")):
    if pending in st.session_state:
        st.session_state[target] = st.session_state.pop(pending)
        st.query_params[target] = st.session_state[target]
        if target == "course":
            st.query_params.pop("session", None)
            st.session_state.pop("session", None)


def sync_course():
    st.query_params["course"] = st.session_state.course
    st.query_params.pop("session", None)
    st.session_state.pop("session", None)


def sync_session():
    st.query_params["session"] = st.session_state.session


def fetch(method: str, path: str, **kwargs):
    """Display API failures in context without erasing persisted data."""
    try:
        return api_request(method, path, **kwargs)
    except RuntimeError as error:
        st.error(str(error))
        st.stop()


def suggest_question(text: str, key: str):
    st.session_state[key] = text


def render_evidence(result: dict):
    chunks = result.get("retrieved_chunks", [])
    if chunks:
        with st.expander(f"查看引用依据 · {len(chunks)} 个片段", icon=":material/library_books:"):
            for item in chunks:
                location = f"{item.get('source', '资料')} · 第 {item.get('page', '?')} 页"
                if item.get("heading"):
                    location += f" · {item['heading']}"
                st.caption(location)
                if item.get("source_deleted"):
                    st.caption("资料已删除 · 以下为历史引用快照，不参与新的检索")
                st.text(item.get("text", ""))
                image = item.get("image_path")
                if image and Path(image).is_file():
                    st.image(image, alt=f"{location}的课件图片")


with st.sidebar:
    st.subheader("课程问答", icon=":material/menu_book:")
    st.caption("我的学习桌 · 资料与对话自动保存")
    connection_slot = st.empty()

try:
    courses = api_request("GET", "/courses")
except RuntimeError as error:
    connection_slot.caption("问答服务暂未连接")
    with st.sidebar:
        st.button("重新连接", icon=":material/refresh:")
    st.title("课程资料智能问答", text_alignment="center")
    st.error(str(error))
    st.caption("启动后端后重新连接，已保存的课程和会话会恢复。")
    st.code("uvicorn api.main:app --reload")
    st.stop()

connection_slot.caption(":material/check_circle: 已连接 · 本地保存")

with st.sidebar:
    st.space("small")
    st.caption("课程资料库")
    if courses:
        course_labels = {course["id"]: course["name"] for course in courses}
        requested = st.query_params.get("course")
        if requested in course_labels:
            st.session_state.course = requested
        course_id = st.selectbox(
            "当前课程", list(course_labels), format_func=course_labels.get,
            key="course", on_change=sync_course, label_visibility="collapsed",
        )
        st.query_params["course"] = course_id
    else:
        course_id = None

    with st.popover("新建课程", icon=":material/add:", width="stretch"):
        with st.form("create_course", clear_on_submit=True):
            course_name = st.text_input("课程名称", placeholder="例如：机器学习", max_chars=100)
            create = st.form_submit_button("创建课程", type="primary", width="stretch")
        if create:
            if not course_name.strip():
                st.warning("请输入课程名称。")
            else:
                course = fetch("POST", "/courses", json={"name": course_name})
                st.session_state.pending_course = course["id"]
                st.session_state.pop("session", None)
                st.rerun()

    if course_id:
        with st.popover("管理课程", icon=":material/folder:", width="stretch"):
            st.caption(f"删除「{course_labels[course_id]}」将移除全部资料、索引和问答会话。")
            if st.button("确认删除当前课程", key="delete_course", width="stretch"):
                fetch("DELETE", f"/courses/{course_id}")
                st.session_state.pending_navigation_reset = "course"
                st.rerun()

if course_id is None:
    st.space("large")
    st.title("为每门课，留一张学习桌", text_alignment="center")
    st.caption("创建课程、上传资料，再开启独立的问答会话。", text_alignment="center")
    with st.container(border=True):
        st.markdown(":material/folder: **课程资料库**")
        st.write("不同课程分别保存资料，同名课件也不会互相覆盖。")
        st.markdown(":material/chat: **独立问答会话**")
        st.write("每次复习可以开启新会话，再次打开时继续之前的讨论。")
    st.stop()

course = next(course for course in courses if course["id"] == course_id)
documents = fetch("GET", f"/courses/{course_id}/documents")
sessions = fetch("GET", f"/courses/{course_id}/sessions")
session_ids = [session["id"] for session in sessions]
if st.session_state.get("session") not in session_ids:
    st.session_state.pop("session", None)

with st.sidebar:
    st.space("small")
    st.caption("问答会话")
    if sessions:
        session_labels = {session["id"]: session["title"] for session in sessions}
        requested = st.query_params.get("session")
        if requested in session_labels:
            st.session_state.session = requested
        session_id = st.selectbox(
            "当前会话", session_ids, format_func=session_labels.get,
            key="session", on_change=sync_session, label_visibility="collapsed",
        )
        st.query_params["session"] = session_id
    else:
        session_id = None
        st.caption("还没有会话，可以先上传资料。")
    if st.button("新建会话", icon=":material/edit_square:", width="stretch"):
        session = fetch("POST", f"/courses/{course_id}/sessions", json={})
        st.session_state.pending_session = session["id"]
        st.rerun()

    st.space("small")
    st.caption("课程资料")
    uploaded = st.file_uploader(
        "添加资料", type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        key=f"uploads_{course_id}_{st.session_state.get(f'upload_generation_{course_id}', 0)}",
        help="上传后立即保存到当前课程。同名文件会更新；首次提问时建立检索索引。",
    )
    changed = False
    for file in uploaded or []:
        digest = hashlib.sha256(file.getvalue()).hexdigest()
        receipt = (course_id, file.name, digest)
        if receipt not in st.session_state.upload_receipts:
            existing = next((doc for doc in documents if doc["name"] == file.name), None)
            if not existing or existing["sha256"] != digest:
                with st.spinner(f"正在保存 {file.name}…"):
                    try:
                        upload_document(course_id, file)
                    except RuntimeError as error:
                        st.error(str(error))
                        st.stop()
                changed = True
            st.session_state.upload_receipts.add(receipt)
    if changed:
        st.rerun()
    st.caption(f"已保存 {len(documents)} 份资料 · 再次打开无需重传")
    if documents:
        with st.popover("管理已上传资料", icon=":material/description:", width="stretch"):
            to_delete = st.selectbox("要删除的资料", [doc["name"] for doc in documents],
                                     key=f"delete_document_name_{course_id}")
            st.caption("删除后所有会话都无法再检索此资料。历史回答保留，引用会标记资料已删除。")
            if st.button("确认删除所选资料", key="delete_document", width="stretch"):
                fetch("DELETE", f"/courses/{course_id}/documents/{quote(to_delete, safe='')}")
                st.session_state.pending_deleted_document = (course_id, to_delete, session_ids)
                generation = f"upload_generation_{course_id}"
                st.session_state[generation] = st.session_state.get(generation, 0) + 1
                st.rerun()

session = fetch("GET", f"/sessions/{session_id}") if session_id else None
doc_names = [doc["name"] for doc in documents]
with st.sidebar:
    if session:
        saved_selection = session["selected_documents"]
        default = doc_names if saved_selection is None else [name for name in saved_selection if name in doc_names]
        selected = st.multiselect(
            "本会话使用的资料", doc_names, default=default,
            key=f"documents_{session_id}", help="每个会话独立记住资料选择。",
        )
        if selected != default:
            fetch("PATCH", f"/sessions/{session_id}/documents", json={"doc_names": selected})
    else:
        selected = doc_names
        for doc in documents:
            st.text(doc["name"])

    with st.expander("问答设置", icon=":material/tune:"):
        show_chunks = st.checkbox("显示引用依据", value=True)
        show_history = st.checkbox("显示完整对话", value=True)
        top_k = st.slider("参考片段数量", 1, 10, min(RERANK_TOP_K, 10))
    if session:
        with st.popover("管理当前会话", icon=":material/more_horiz:", width="stretch"):
            st.caption("仅删除当前会话的问答记录，课程资料会保留。")
            if st.button("确认清空", key="clear_turns", width="stretch"):
                fetch("DELETE", f"/sessions/{session_id}/turns")
                st.rerun()
            st.caption("删除会话会移除该会话及历史记录，课程资料保留。")
            if st.button("确认删除当前会话", key="delete_session", width="stretch"):
                fetch("DELETE", f"/sessions/{session_id}")
                st.session_state.pending_navigation_reset = "session"
                st.rerun()

st.caption(f"{course['name']} / {session['title'] if session else '课程资料库'}")
if not session:
    st.title(course["name"])
    st.caption("资料已保存在这门课程中。开启一个新会话，开始复习。")
    if not documents:
        st.info("从左侧添加课件、讲义或笔记，上传后立即保存。", icon=":material/upload_file:")
    else:
        with st.container(border=True):
            for doc in documents:
                st.text(doc["name"])
    st.stop()

turns = session["turns"]
question_key = f"question_{session_id}"
if not turns:
    st.space("medium")
    st.title("今天，想弄懂什么？", text_alignment="center")
    st.caption("围绕课程资料提问，让每个答案都有据可查。", text_alignment="center")
    st.caption(f"本会话使用 {len(selected)} 份资料 · 对话自动保存", text_alignment="center")
    examples = [
        ("总结课件", "请总结这些课件的主要内容。"),
        ("梳理知识点", "请提炼这些资料的核心知识点，并说明它们之间的联系。"),
        ("解释概念", "请解释资料中的关键概念，并给出对应页码。"),
        ("制定复习提纲", "请根据这些资料整理一份复习提纲。"),
    ]
    columns = st.columns(2)
    for index, (label, text) in enumerate(examples):
        with columns[index % 2]:
            st.button(label, key=f"suggestion_{index}", width="stretch",
                      on_click=suggest_question, args=(text, question_key))
    if not selected:
        st.caption("先在左侧上传资料并选择本会话的知识来源。")
else:
    for turn in turns if show_history else turns[-1:]:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        result = turn["result"]
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            if result.get("retries"):
                st.caption(f"查询经过 {result['retries']} 次改写后重新检索。")
            if show_chunks:
                render_evidence(result)

with st.container():
    question = st.chat_input(
        "向课程资料提问…", key=question_key, disabled=not selected, submit_mode="disable",
    )
if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("正在查阅课程资料…"):
            try:
                api_request(
                    "POST", "/chat", json={"session_id": session_id, "question": question,
                                             "doc_names": selected, "top_k": top_k}, timeout=600,
                )
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.rerun()
