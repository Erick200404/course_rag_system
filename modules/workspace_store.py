"""Persistent course libraries and independent chat sessions (single-user app)."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import tempfile
import time
from uuid import uuid4

WORKSPACE_DB = Path("storage/workspace.db")
COURSE_DATA_ROOT = Path("data/courses")
COURSE_STORAGE_ROOT = Path("storage/courses")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@contextmanager
def database():
    WORKSPACE_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(WORKSPACE_DB), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS courses (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS course_documents (
            course_id TEXT NOT NULL REFERENCES courses(id),
            name TEXT NOT NULL, sha256 TEXT NOT NULL, size INTEGER NOT NULL,
            updated_at REAL NOT NULL, PRIMARY KEY(course_id, name)
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id),
            title TEXT NOT NULL, selected_documents TEXT,
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            question TEXT NOT NULL, result TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, id);
    """)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def validate_filename(name: str) -> str:
    if (not name or name in (".", "..") or any(c in name for c in '/\\:\x00')
            or name.endswith((" ", ".")) or Path(name).name != name):
        raise ValueError("文件名不能包含路径或非法字符")
    return name


def _required(db, table: str, item_id: str) -> dict:
    row = db.execute(f"SELECT * FROM {table} WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        raise LookupError("课程或会话不存在")
    return dict(row)


def create_course(name: str) -> dict:
    name = name.strip()
    if not name or len(name) > 100:
        raise ValueError("课程名称需为 1–100 个字符")
    record = {"id": uuid4().hex, "name": name, "created_at": time.time()}
    with database() as db:
        db.execute("INSERT INTO courses VALUES (:id, :name, :created_at)", record)
    return record


def list_courses() -> list[dict]:
    with database() as db:
        return [dict(row) for row in db.execute("SELECT * FROM courses ORDER BY created_at, id")]


def get_course(course_id: str) -> dict:
    with database() as db:
        return _required(db, "courses", course_id)


def course_storage_dir(course_id: str) -> Path:
    get_course(course_id)  # Only server-generated, existing IDs reach filesystem paths.
    return COURSE_STORAGE_ROOT / course_id


def list_course_documents(course_id: str) -> list[dict]:
    with database() as db:
        _required(db, "courses", course_id)
        return [dict(row) for row in db.execute(
            "SELECT * FROM course_documents WHERE course_id = ? ORDER BY name", (course_id,)
        )]


def save_course_document(course_id: str, name: str, content: bytes) -> dict:
    get_course(course_id)
    validate_filename(name)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("文件不能为空，且不能超过 200MB")
    directory = COURSE_DATA_ROOT / course_id
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(dir=directory, suffix=".uploading")
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
        os.replace(temporary_path, directory / name)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    record = {"course_id": course_id, "name": name,
              "sha256": hashlib.sha256(content).hexdigest(),
              "size": len(content), "updated_at": time.time()}
    with database() as db:
        db.execute("""INSERT INTO course_documents VALUES
            (:course_id, :name, :sha256, :size, :updated_at)
            ON CONFLICT(course_id, name) DO UPDATE SET sha256=excluded.sha256,
            size=excluded.size, updated_at=excluded.updated_at""", record)
    return record


def resolve_documents(course_id: str, names: list[str]) -> list[str]:
    available = {doc["name"] for doc in list_course_documents(course_id)}
    paths = []
    for name in dict.fromkeys(names):
        validate_filename(name)
        if name not in available:
            raise ValueError(f"资料不属于当前课程：{name}")
        path = COURSE_DATA_ROOT / course_id / name
        if not path.is_file():
            raise FileNotFoundError(f"资料文件缺失：{name}")
        paths.append(str(path))
    return paths


def create_session(course_id: str, title: str = "新会话") -> dict:
    title = title.strip() or "新会话"
    if len(title) > 100:
        raise ValueError("会话标题不能超过 100 个字符")
    now = time.time()
    with database() as db:
        _required(db, "courses", course_id)
        record = {"id": uuid4().hex, "course_id": course_id, "title": title,
                  "created_at": now, "updated_at": now}
        db.execute("""INSERT INTO sessions
            (id, course_id, title, created_at, updated_at)
            VALUES (:id, :course_id, :title, :created_at, :updated_at)""", record)
    return get_session(record["id"])


def _session(row) -> dict:
    record = dict(row)
    raw = record["selected_documents"]
    record["selected_documents"] = json.loads(raw) if raw is not None else None
    return record


def list_sessions(course_id: str) -> list[dict]:
    with database() as db:
        _required(db, "courses", course_id)
        return [_session(row) for row in db.execute(
            "SELECT * FROM sessions WHERE course_id = ? ORDER BY created_at, id", (course_id,)
        )]


def get_session(session_id: str) -> dict:
    with database() as db:
        record = _session(_required(db, "sessions", session_id))
        record["turns"] = [
            {"id": row["id"], "question": row["question"],
             "result": json.loads(row["result"]), "created_at": row["created_at"]}
            for row in db.execute("SELECT * FROM turns WHERE session_id = ? ORDER BY id", (session_id,))
        ]
    return record


def select_session_documents(session_id: str, names: list[str]) -> None:
    session = get_session(session_id)
    resolve_documents(session["course_id"], names)
    with database() as db:
        db.execute("UPDATE sessions SET selected_documents = ?, updated_at = ? WHERE id = ?",
                   (json.dumps(list(dict.fromkeys(names)), ensure_ascii=False), time.time(), session_id))


def append_turn(session_id: str, question: str, result: dict) -> None:
    now = time.time()
    with database() as db:
        session = _required(db, "sessions", session_id)
        db.execute("INSERT INTO turns (session_id, question, result, created_at) VALUES (?, ?, ?, ?)",
                   (session_id, question, json.dumps(result, ensure_ascii=False), now))
        title = question[:40] if session["title"] == "新会话" else session["title"]
        db.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, now, session_id))


def clear_turns(session_id: str) -> None:
    with database() as db:
        _required(db, "sessions", session_id)
        db.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))


def remove_owned_tree(root: Path, target: Path) -> None:
    """Delete only a verified child directory; never traverse a linked target."""
    resolved_root, resolved_target = root.resolve(), target.resolve()
    if target.is_symlink() or resolved_target == resolved_root or not resolved_target.is_relative_to(resolved_root):
        raise ValueError("删除路径超出课程存储范围")
    if target.exists():
        shutil.rmtree(resolved_target)


def delete_course_document(course_id: str, name: str) -> None:
    get_course(course_id)
    validate_filename(name)
    with database() as db:
        document = db.execute("SELECT name FROM course_documents WHERE course_id = ? AND name = ?",
                              (course_id, name)).fetchone()
        if not document:
            raise LookupError("资料不存在")
        directory = COURSE_DATA_ROOT / course_id
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("删除路径超出课程存储范围")
        path.unlink(missing_ok=True)
        db.execute("DELETE FROM course_documents WHERE course_id = ? AND name = ?", (course_id, name))
        for row in db.execute("SELECT id, selected_documents FROM sessions WHERE course_id = ?", (course_id,)):
            if row["selected_documents"] is not None:
                selected = [item for item in json.loads(row["selected_documents"]) if item != name]
                db.execute("UPDATE sessions SET selected_documents = ? WHERE id = ?",
                           (json.dumps(selected, ensure_ascii=False), row["id"]))
        # Preserve historical answers, but mark removed evidence and stop showing
        # its cached images. A later file with the same name is a new document.
        for row in db.execute("""SELECT turns.id, turns.result FROM turns JOIN sessions
                                ON turns.session_id = sessions.id WHERE sessions.course_id = ?""", (course_id,)):
            result = json.loads(row["result"])
            changed = False
            for chunk in result.get("retrieved_chunks", []):
                if chunk.get("source") == name:
                    chunk["source_deleted"] = True
                    chunk.pop("image_path", None)
                    changed = True
            if changed:
                db.execute("UPDATE turns SET result = ? WHERE id = ?",
                           (json.dumps(result, ensure_ascii=False), row["id"]))


def delete_session(session_id: str) -> None:
    with database() as db:
        _required(db, "sessions", session_id)
        db.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def delete_course(course_id: str) -> None:
    get_course(course_id)
    remove_owned_tree(COURSE_DATA_ROOT, COURSE_DATA_ROOT / course_id)
    with database() as db:
        db.execute("DELETE FROM turns WHERE session_id IN (SELECT id FROM sessions WHERE course_id = ?)", (course_id,))
        db.execute("DELETE FROM sessions WHERE course_id = ?", (course_id,))
        db.execute("DELETE FROM course_documents WHERE course_id = ?", (course_id,))
        db.execute("DELETE FROM courses WHERE id = ?", (course_id,))

