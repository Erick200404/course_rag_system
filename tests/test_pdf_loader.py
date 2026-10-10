"""
pdf_loader 单元测试：markdown 后端（pymupdf4llm）与 text 后端。
"""

import fitz

from modules.pdf_loader import load_pdf


def _make_pdf(path, lines):
    """
    生成一个简单的测试 PDF：
    第一行为大字号标题（pymupdf4llm 会识别为 Markdown 标题），
    其余为正文字号。
    """
    doc = fitz.open()

    page = doc.new_page()

    y = 100
    for text, fontsize in lines:
        page.insert_text((72, y), text, fontsize=fontsize)
        y += fontsize + 20

    doc.save(str(path))
    doc.close()

    return path


class TestValidation:
    def test_missing_file_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_pdf(str(tmp_path / "not_exist.pdf"))

    def test_non_pdf_raises(self, tmp_path):
        import pytest

        txt = tmp_path / "note.txt"
        txt.write_text("hello")

        with pytest.raises(ValueError):
            load_pdf(str(txt))


class TestTextBackend:
    def test_basic_structure(self, tmp_path):
        pdf = _make_pdf(tmp_path / "t.pdf", [("Hello Title", 18), ("Body text here", 11)])

        pages = load_pdf(str(pdf), backend="text")

        assert len(pages) == 1
        assert pages[0]["page"] == 1
        assert pages[0]["source"] == "t.pdf"
        assert "Hello Title" in pages[0]["text"]


class TestMarkdownBackend:
    def test_basic_structure(self, tmp_path):
        pdf = _make_pdf(tmp_path / "t.pdf", [("Hello Title", 18), ("Body text here", 11)])

        pages = load_pdf(str(pdf), backend="markdown")

        assert len(pages) == 1
        assert pages[0]["page"] == 1
        assert pages[0]["source"] == "t.pdf"
        # 正文应当被提取出来
        assert "Body text here" in pages[0]["text"]

    def test_multiple_pages(self, tmp_path):
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 100), f"Page {i + 1} body content", fontsize=11)
        pdf_path = tmp_path / "multi.pdf"
        doc.save(str(pdf_path))
        doc.close()

        pages = load_pdf(str(pdf_path), backend="markdown")

        assert len(pages) == 3
        assert [p["page"] for p in pages] == [1, 2, 3]

    def test_loader_to_splitter_integration(self, tmp_path):
        """
        端到端：markdown 加载 -> 标题切分，
        若 pymupdf4llm 识别出标题，则 chunk 应带 heading。
        """
        from modules.text_splitter import split_text

        pdf = _make_pdf(
            tmp_path / "t.pdf",
            [("Chapter Title", 22), ("Body text here", 11)],
        )

        pages = load_pdf(str(pdf), backend="markdown")
        chunks = split_text(pages, backend="markdown")

        assert len(chunks) >= 1
        # 无论标题是否被识别为 #，正文都应进入 chunk
        assert any("Body text here" in c["text"] for c in chunks)
