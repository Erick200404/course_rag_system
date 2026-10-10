"""
doc_registry 单元测试：内容 hash 知识库 id 与文档登记。
"""

from modules.doc_registry import (
    compute_kb_id,
    file_sha256,
    list_documents,
    register_documents,
    remove_document,
)


def _make_pdf(path, content: bytes):
    path.write_bytes(content)
    return path


class TestComputeKbId:
    def test_content_change_changes_kb_id(self, tmp_path):
        pdf = _make_pdf(tmp_path / "a.pdf", b"hello world")

        id_before = compute_kb_id([str(pdf)], "hf", "bge-small-zh")

        # 文件内容更新
        _make_pdf(pdf, b"hello world, updated")

        id_after = compute_kb_id([str(pdf)], "hf", "bge-small-zh")

        assert id_before != id_after

    def test_order_insensitive(self, tmp_path):
        a = _make_pdf(tmp_path / "a.pdf", b"content-a")
        b = _make_pdf(tmp_path / "b.pdf", b"content-b")

        id_1 = compute_kb_id([str(a), str(b)], "hf", "m")
        id_2 = compute_kb_id([str(b), str(a)], "hf", "m")

        assert id_1 == id_2

    def test_embedding_model_change_changes_kb_id(self, tmp_path):
        pdf = _make_pdf(tmp_path / "a.pdf", b"same content")

        id_1 = compute_kb_id([str(pdf)], "hf", "model-1")
        id_2 = compute_kb_id([str(pdf)], "hf", "model-2")

        assert id_1 != id_2

    def test_index_schema_change_changes_kb_id(self, tmp_path):
        pdf = _make_pdf(tmp_path / "a.pdf", b"same content")

        id_1 = compute_kb_id([str(pdf)], "hf", "m", index_schema_version="v1")
        id_2 = compute_kb_id([str(pdf)], "hf", "m", index_schema_version="v2")

        assert id_1 != id_2

    def test_different_file_sets_differ(self, tmp_path):
        a = _make_pdf(tmp_path / "a.pdf", b"content-a")
        b = _make_pdf(tmp_path / "b.pdf", b"content-b")

        assert compute_kb_id([str(a)], "hf", "m") != compute_kb_id([str(b)], "hf", "m")

    def test_kb_id_is_hex_digest(self, tmp_path):
        pdf = _make_pdf(tmp_path / "a.pdf", b"x")

        kb_id = compute_kb_id([str(pdf)], "hf", "m")

        assert len(kb_id) == 32
        int(kb_id, 16)  # 不抛异常即为合法十六进制


class TestRegistry:
    def test_register_and_list(self, tmp_path):
        db = str(tmp_path / "registry.db")
        pdf = _make_pdf(tmp_path / "a.pdf", b"pdf content")

        register_documents([str(pdf)], db_path=db)

        docs = list_documents(db_path=db)
        assert len(docs) == 1
        assert docs[0]["filename"] == "a.pdf"
        assert docs[0]["sha256"] == file_sha256(str(pdf))
        assert docs[0]["size"] == len(b"pdf content")

    def test_reregister_updates_hash(self, tmp_path):
        db = str(tmp_path / "registry.db")
        pdf = _make_pdf(tmp_path / "a.pdf", b"version-1")

        register_documents([str(pdf)], db_path=db)
        _make_pdf(pdf, b"version-2")
        register_documents([str(pdf)], db_path=db)

        docs = list_documents(db_path=db)
        assert len(docs) == 1
        assert docs[0]["sha256"] == file_sha256(str(pdf))

    def test_remove_document(self, tmp_path):
        db = str(tmp_path / "registry.db")
        pdf = _make_pdf(tmp_path / "a.pdf", b"content")

        register_documents([str(pdf)], db_path=db)
        remove_document("a.pdf", db_path=db)

        assert list_documents(db_path=db) == []
