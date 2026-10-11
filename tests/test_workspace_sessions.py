"""Course isolation and durable history, without model calls."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from modules import workspace_store as workspace


@pytest.fixture
def workspace_env(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "WORKSPACE_DB", tmp_path / "workspace.db")
    monkeypatch.setattr(workspace, "COURSE_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(workspace, "COURSE_STORAGE_ROOT", tmp_path / "storage")
    api_main._course_stores.clear()
    api_main._retriever_cache.clear()
    yield TestClient(api_main.app)
    api_main._course_stores.clear()
    api_main._retriever_cache.clear()


def create(client, course_name="机器学习"):
    course = client.post("/courses", json={"name": course_name}).json()
    session = client.post(f"/courses/{course['id']}/sessions", json={}).json()
    return course, session


def upload(client, course, name="课件.txt", body=b"course content"):
    return client.put(f"/courses/{course['id']}/documents/{name}", content=body)


def test_same_name_documents_are_isolated_and_survive_reopen(workspace_env):
    client = workspace_env
    first, _ = create(client, "机器学习")
    second, _ = create(client, "数据库")
    assert upload(client, first, body=b"machine learning").status_code == 200
    assert upload(client, second, body=b"database").status_code == 200
    paths = [workspace.resolve_documents(course["id"], ["课件.txt"])[0] for course in (first, second)]
    assert Path(paths[0]).read_bytes() == b"machine learning"
    assert Path(paths[1]).read_bytes() == b"database"
    # Fresh client/SQLite connection, no session_state to restore from.
    reopened = TestClient(api_main.app)
    assert len(reopened.get("/courses").json()) == 2
    assert reopened.get(f"/courses/{first['id']}/documents").json()[0]["name"] == "课件.txt"
    assert upload(client, first, body=b"updated").status_code == 200
    assert Path(paths[0]).read_bytes() == b"updated"
    assert Path(paths[1]).read_bytes() == b"database"


def test_chat_uses_only_server_session_history_and_persists_result(workspace_env, monkeypatch):
    client = workspace_env
    course, session = create(client)
    other = client.post(f"/courses/{course['id']}/sessions", json={}).json()
    upload(client, course)
    workspace.append_turn(other["id"], "另一个会话的问题", {"answer": "另一个答案"})
    captured = []
    monkeypatch.setattr(api_main, "build_rag_pipeline",
                        lambda paths, course_id=None: (None, [], course_id, "retriever"))
    monkeypatch.setattr(api_main, "RAG_FLOW", "crag")

    def crag(**kwargs):
        captured.append(kwargs)
        return {"question": kwargs["question"], "original_question": kwargs["question"],
                "chunks": [], "answer": "课程答案", "retries": 0, "fallback": False}

    monkeypatch.setattr(api_main, "run_crag", crag)
    payload = {"session_id": session["id"], "question": "什么是 RAG？",
               "history": [{"question": "伪造的跨会话历史", "answer": "不应使用"}]}
    assert client.post("/chat", json=payload).status_code == 200
    assert captured[0]["history"] == []
    payload["question"] = "它有什么用途？"
    assert client.post("/chat", json=payload).status_code == 200
    assert captured[1]["history"] == [{"question": "什么是 RAG？", "answer": "课程答案"}]
    saved = TestClient(api_main.app).get(f"/sessions/{session['id']}").json()
    assert len(saved["turns"]) == 2
    assert saved["turns"][1]["result"]["answer"] == "课程答案"
    assert len(client.get(f"/sessions/{other['id']}").json()["turns"]) == 1
    assert client.delete(f"/sessions/{session['id']}/turns").status_code == 200
    assert client.get(f"/sessions/{session['id']}").json()["turns"] == []
    assert len(client.get(f"/sessions/{other['id']}").json()["turns"]) == 1
    assert client.get(f"/courses/{course['id']}/documents").json()


def test_document_selection_is_durable_and_rejects_another_course(workspace_env):
    client = workspace_env
    course, first = create(client)
    other, _ = create(client, "另一门课")
    second = client.post(f"/courses/{course['id']}/sessions", json={}).json()
    upload(client, course, "a.txt")
    upload(client, course, "b.txt")
    upload(client, other, "private.txt")
    assert client.patch(f"/sessions/{first['id']}/documents", json={"doc_names": ["a.txt"]}).status_code == 200
    assert client.patch(f"/sessions/{second['id']}/documents", json={"doc_names": []}).status_code == 200
    assert client.get(f"/sessions/{first['id']}").json()["selected_documents"] == ["a.txt"]
    assert client.get(f"/sessions/{second['id']}").json()["selected_documents"] == []
    response = client.patch(f"/sessions/{first['id']}/documents", json={"doc_names": ["private.txt"]})
    assert response.status_code == 400
    assert client.post("/chat", json={"session_id": second["id"], "question": "问答"}).status_code == 400


@pytest.mark.parametrize("name", ["../secret.txt", "..\\secret.txt", "C:secret.txt", "bad.txt."])
def test_upload_paths_are_rejected(workspace_env, name):
    course, _ = create(workspace_env)
    with pytest.raises(ValueError):
        workspace.save_course_document(course["id"], name, b"bad")


def test_invalid_courses_empty_upload_and_unsupported_format(workspace_env):
    client = workspace_env
    course, _ = create(client)
    assert client.get("/sessions/missing").status_code == 404
    assert client.put("/courses/missing/documents/a.txt", content=b"x").status_code == 404
    assert upload(client, course, body=b"").status_code == 400
    assert upload(client, course, name="a.exe").status_code == 400


def test_chroma_course_stores_are_separate_even_for_same_names(workspace_env, monkeypatch):
    client = workspace_env
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    first, _ = create(client, "机器学习")
    second, _ = create(client, "数据库")
    body_a = "# 第一章\n\n机器学习课程介绍模型训练方法与检索增强生成技术的基本原理和应用。".encode()
    body_b = "# 第一章\n\n数据库课程介绍向量索引和数据库查询方法的基本原理以及存储系统。".encode()
    upload(client, first, "lecture.md", body_a)
    upload(client, second, "lecture.md", body_b)
    monkeypatch.setattr(api_main, "embed_chunks", lambda chunks: [
        {**chunk, "embedding": [1., 0., 0.]} for chunk in chunks])
    first_paths = workspace.resolve_documents(first["id"], ["lecture.md"])
    second_paths = workspace.resolve_documents(second["id"], ["lecture.md"])
    _, first_meta, first_id, _ = api_main.build_rag_pipeline(first_paths, first["id"])
    _, second_meta, second_id, _ = api_main.build_rag_pipeline(second_paths, second["id"])
    assert first_id != second_id
    assert "机器学习" in first_meta[0]["text"]
    assert "数据库" in second_meta[0]["text"]
    upload(client, second, "lecture.md", body_b + "更新".encode())
    api_main.build_rag_pipeline(second_paths, second["id"])
    _, first_again, _, _ = api_main.build_rag_pipeline(first_paths, first["id"])
    assert first_again == first_meta


def test_identical_content_different_names_respects_selected_sources(workspace_env, monkeypatch):
    course, _ = create(workspace_env)
    content = "# 检索\n\n检索增强生成技术通过课程资料中的相关知识构造回答，提供内容溯源与依据。".encode()
    upload(workspace_env, course, "a.md", content)
    upload(workspace_env, course, "b.md", content)
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setattr(api_main, "embed_chunks", lambda chunks: [
        {**chunk, "embedding": [1., 0., 0.]} for chunk in chunks])
    paths = workspace.resolve_documents(course["id"], ["a.md", "b.md"])
    api_main.build_rag_pipeline(paths, course["id"])
    view, metadata, _, _ = api_main.build_rag_pipeline(paths[:1], course["id"])
    assert {chunk["source"] for chunk in metadata} == {"a.md"}
    assert {chunk["source"] for chunk in view.search([1., 0., 0.], 5)} == {"a.md"}
