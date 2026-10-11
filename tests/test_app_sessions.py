"""Exercise real Streamlit widgets against a persistent test backend."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import api.main as api_main
from modules import api_client
from modules import workspace_store as workspace

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def ui_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "WORKSPACE_DB", tmp_path / "workspace.db")
    monkeypatch.setattr(workspace, "COURSE_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(workspace, "COURSE_STORAGE_ROOT", tmp_path / "storage")
    client = TestClient(api_main.app)

    def call(method, path, **kwargs):
        kwargs.pop("timeout", None)
        response = client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(response.json()["detail"])
        return response.json()

    monkeypatch.setattr(api_client, "api_request", call)
    first = workspace.create_course("机器学习")
    second = workspace.create_course("数据库")
    for course in (first, second):
        workspace.save_course_document(course["id"], "课件.txt", b"lecture content")
    first_session = workspace.create_session(first["id"])
    second_session = workspace.create_session(second["id"])
    workspace.append_turn(first_session["id"], "机器学习的问题", {
        "answer": "机器学习的回答", "retrieved_chunks": [], "flow": "simple"})
    workspace.append_turn(second_session["id"], "数据库的问题", {
        "answer": "数据库的回答", "retrieved_chunks": [], "flow": "simple"})
    return first, second, first_session, second_session


def test_reopen_and_switch_course_restore_only_their_own_history(ui_backend):
    first, second, first_session, second_session = ui_backend
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not app.exception
    assert app.selectbox(key="course").value == first["id"]
    assert app.multiselect(key=f"documents_{first_session['id']}").value == ["课件.txt"]
    assert [message.markdown[0].value for message in app.chat_message] == ["机器学习的问题", "机器学习的回答"]
    app.selectbox(key="course").set_value(second["id"]).run()
    assert not app.exception
    assert app.selectbox(key="session").value == second_session["id"]
    assert [message.markdown[0].value for message in app.chat_message] == ["数据库的问题", "数据库的回答"]
    reopened = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not reopened.exception
    assert reopened.chat_message[0].markdown[0].value == "机器学习的问题"


def test_new_session_and_empty_document_selection_survive_reload(ui_backend):
    first, _, old, _ = ui_backend
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    next(button for button in app.button if button.label == "新建会话").click().run()
    assert not app.exception
    new_id = app.selectbox(key="session").value
    assert new_id != old["id"]
    assert not app.chat_message
    app.multiselect(key=f"documents_{new_id}").set_value([]).run()
    assert not app.exception
    assert workspace.get_session(new_id)["selected_documents"] == []
    # A different browser session loads the same saved conversation.
    reopened = AppTest.from_file(APP_PATH, default_timeout=15)
    reopened.query_params["course"] = first["id"]
    reopened.query_params["session"] = new_id
    reopened.run()
    assert not reopened.exception
    assert reopened.selectbox(key="session").value == new_id
    assert reopened.multiselect(key=f"documents_{new_id}").value == []
    assert reopened.chat_input[0].disabled


def test_clear_history_keeps_documents_and_other_session(ui_backend):
    _, _, first, second = ui_backend
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    app.button(key="clear_turns").click().run()
    assert not app.exception
    assert not app.chat_message
    assert workspace.get_session(first["id"])["turns"] == []
    assert workspace.get_session(second["id"])["turns"]
    assert app.multiselect(key=f"documents_{first['id']}").value == ["课件.txt"]


def test_suggestion_fills_input_and_chat_saved_in_backend(ui_backend, monkeypatch):
    monkeypatch.setattr(api_main, "RAG_FLOW", "simple")
    monkeypatch.setattr(api_main, "build_rag_pipeline", lambda paths, course_id=None: (None, [], "kb", object()))
    monkeypatch.setattr(api_main, "generate_answer", lambda **kwargs: {
        "query": kwargs["query"], "answer": "生成的答案", "retrieved_chunks": []})
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    next(button for button in app.button if button.label == "新建会话").click().run()
    session_id = app.selectbox(key="session").value
    app.button(key="suggestion_0").click().run()
    assert not app.exception
    app.chat_input[0].set_value("请总结课件").run()
    assert not app.exception
    assert app.chat_message[-1].markdown[0].value == "生成的答案"
    assert workspace.get_session(session_id)["turns"][0]["question"] == "请总结课件"


def test_offline_service_preserves_reconnect_ui(monkeypatch):
    def offline(*args, **kwargs):
        raise RuntimeError("无法连接问答服务")

    monkeypatch.setattr(api_client, "api_request", offline)
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not app.exception
    assert app.error
    assert any(button.label == "重新连接" for button in app.button)


def test_course_creation_switches_to_new_empty_library(ui_backend):
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    next(widget for widget in app.text_input if widget.label == "课程名称").set_value("操作系统")
    next(button for button in app.button if button.label == "创建课程").click().run()
    assert not app.exception
    new_id = app.selectbox(key="course").value
    assert workspace.get_course(new_id)["name"] == "操作系统"
    assert app.query_params["course"] == new_id
    assert not app.chat_message
    assert "还没有会话" in " ".join(caption.value for caption in app.caption)


def test_selected_session_stays_in_url_when_automatic_title_changes(ui_backend, monkeypatch):
    monkeypatch.setattr(api_main, "RAG_FLOW", "simple")
    monkeypatch.setattr(api_main, "build_rag_pipeline", lambda paths, course_id=None: (None, [], "kb", object()))
    monkeypatch.setattr(api_main, "generate_answer", lambda **kwargs: {
        "query": kwargs["query"], "answer": "新回答", "retrieved_chunks": []})
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    next(button for button in app.button if button.label == "新建会话").click().run()
    session_id = app.selectbox(key="session").value
    app.chat_input[0].set_value("这是用来更新会话标题的第一问").run()
    assert not app.exception
    assert app.query_params["session"] == session_id
    restored = AppTest.from_file(APP_PATH, default_timeout=15)
    restored.query_params.update(app.query_params)
    restored.run()
    assert not restored.exception
    assert restored.selectbox(key="session").value == session_id
    assert restored.chat_message[-1].markdown[0].value == "新回答"
