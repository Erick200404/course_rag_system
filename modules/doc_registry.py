"""
SQLite 文档注册表。

职责：
1. 记录每个已入库文档的 sha256 / 大小 / 修改时间
2. 基于文件内容 hash 计算知识库 id（kb_id）：
   - 文件内容或 embedding 配置变化 -> 新 kb_id -> 自动重建索引
   - 解决"同名文件更新后仍复用旧索引"的问题
3. 提供文档的登记 / 查询 / 删除接口，为后续增量更新打基础
"""

import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List

from config import INDEX_SCHEMA_VERSION

# 注册表默认存放位置
REGISTRY_PATH = os.path.join("storage", "registry.db")

DocumentRecord = Dict[str, Any]

_CREATE_DOCUMENTS_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    filename      TEXT PRIMARY KEY,
    path          TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    size          INTEGER NOT NULL,
    mtime         REAL NOT NULL,
    registered_at REAL NOT NULL
)
"""


@contextmanager
def _db(db_path: str):
    """
    打开数据库连接的上下文管理器：正常退出自动提交，异常回滚。
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_CREATE_DOCUMENTS_SQL)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def file_sha256(path: str, block_size: int = 1024 * 1024) -> str:
    """
    计算文件的 sha256（流式读取，避免大文件一次性读入内存）。
    """
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def register_documents(
    pdf_paths: List[str],
    db_path: str = REGISTRY_PATH,
) -> List[DocumentRecord]:
    """
    将一组 PDF 文件登记进注册表（存在则更新）。

    返回:
        每个文件的登记信息列表
    """
    now = time.time()
    records: List[DocumentRecord] = []

    with _db(db_path) as conn:
        for path in pdf_paths:
            record = {
                "filename": os.path.basename(path),
                "path": os.path.abspath(path),
                "sha256": file_sha256(path),
                "size": os.path.getsize(path),
                "mtime": os.path.getmtime(path),
                "registered_at": now,
            }

            # 同名文件再次登记时更新全部字段
            conn.execute(
                """
                INSERT INTO documents
                    (filename, path, sha256, size, mtime, registered_at)
                VALUES
                    (:filename, :path, :sha256, :size, :mtime, :registered_at)
                ON CONFLICT(filename) DO UPDATE SET
                    path = excluded.path,
                    sha256 = excluded.sha256,
                    size = excluded.size,
                    mtime = excluded.mtime,
                    registered_at = excluded.registered_at
                """,
                record,
            )

            records.append(record)

    return records


def list_documents(db_path: str = REGISTRY_PATH) -> List[DocumentRecord]:
    """
    列出注册表中的全部文档。
    """
    columns = ("filename", "path", "sha256", "size", "mtime", "registered_at")

    with _db(db_path) as conn:
        rows = conn.execute(
            f"SELECT {', '.join(columns)} FROM documents ORDER BY filename"
        ).fetchall()

    return [dict(zip(columns, row)) for row in rows]


def remove_document(filename: str, db_path: str = REGISTRY_PATH) -> None:
    """
    从注册表中删除指定文档。
    """
    with _db(db_path) as conn:
        conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))


def compute_kb_id(
    pdf_paths: List[str],
    embedding_backend: str,
    embedding_model: str,
    index_schema_version: str = INDEX_SCHEMA_VERSION,
) -> str:
    """
    根据多个 PDF 的内容 hash 和 embedding 配置生成知识库 id。

    与旧版（只 hash 文件名）的区别：
    - 任何 PDF 文件内容发生变化 -> kb_id 改变 -> 重建索引
    - 切换 embedding backend / 模型 -> kb_id 改变 -> 重建索引
    - 上传顺序不同 -> kb_id 不变（先按文件名排序）

    返回:
        32 位 md5 十六进制字符串，作为 storage 下的子目录名
    """
    # 按文件名排序，保证同一组文件与上传顺序无关
    sorted_paths = sorted(pdf_paths, key=lambda p: os.path.basename(p).lower())

    # 拼接 "文件名:内容hash"
    parts = [
        f"{os.path.basename(path)}:{file_sha256(path)}"
        for path in sorted_paths
    ]

    # 知识库标识 = 文件集合内容 + embedding 配置
    raw_kb_key = "||".join(parts)
    raw_kb_key += f"||backend={embedding_backend}||model={embedding_model}"
    raw_kb_key += f"||index_schema={index_schema_version}"

    return hashlib.md5(raw_kb_key.encode("utf-8")).hexdigest()
