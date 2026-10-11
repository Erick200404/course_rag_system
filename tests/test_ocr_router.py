"""
ocr_router 单元测试：扫描页检测、OCR 路由与降级行为。

OCR 引擎用 monkeypatch 伪造，不依赖 rapidocr_onnxruntime 与模型下载。
"""

import fitz
import pytest

import modules.ocr_router as ocr_router
import modules.pdf_loader as pdf_loader
from modules.ocr_router import (
    extract_pdf_pages_with_ocr,
    is_scanned_page,
    ocr_page,
    text_density,
)

CJK_FONT = "china-s"


def _text_pdf(path, lines):
    """生成带文字层的 PDF。"""
    doc = fitz.open()
    page = doc.new_page()
    y = 100
    for text, fontsize in lines:
        page.insert_text((72, y), text, fontsize=fontsize, fontname=CJK_FONT)
        y += fontsize + 24
    doc.save(str(path))
    doc.close()
    return str(path)


def _scanned_pdf(path):
    """生成扫描页 PDF：把文字画成矢量路径（无文字层），模拟图片页。"""
    doc = fitz.open()
    page = doc.new_page()

    # 用绘图指令画一些内容（有像素但无文本层）
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(72, 72, 540, 400))
    shape.finish(color=(0, 0, 0), width=2)
    shape.commit()

    doc.save(str(path))
    doc.close()
    return str(path)


class TestScannedPageDetection:
    def test_text_page_is_not_scanned(self, tmp_path):
        path = _text_pdf(
            tmp_path / "text.pdf",
            [("第一章 检索增强生成", 20), ("本章系统介绍检索增强生成技术在课程资料问答中的应用方式与工程实践要点。", 11)],
        )

        doc = fitz.open(path)
        try:
            assert is_scanned_page(doc[0]) is False
        finally:
            doc.close()

    def test_empty_page_is_scanned(self, tmp_path):
        path = _scanned_pdf(tmp_path / "scan.pdf")

        doc = fitz.open(path)
        try:
            assert is_scanned_page(doc[0]) is True
        finally:
            doc.close()

    def test_text_density_ordering(self, tmp_path):
        text_path = _text_pdf(
            tmp_path / "text.pdf",
            [("正文" * 60, 11)],
        )
        scan_path = _scanned_pdf(tmp_path / "scan.pdf")

        text_doc = fitz.open(text_path)
        scan_doc = fitz.open(scan_path)
        try:
            assert text_density(text_doc[0]) > text_density(scan_doc[0])
        finally:
            text_doc.close()
            scan_doc.close()


class TestOcrRouting:
    def test_scanned_page_uses_ocr_when_available(self, tmp_path, monkeypatch):
        path = _scanned_pdf(tmp_path / "scan.pdf")

        monkeypatch.setattr(ocr_router, "OCR_ENABLED", True)
        monkeypatch.setattr(
            ocr_router,
            "ocr_page",
            lambda page: "OCR 提取出来的扫描页文字内容",
        )

        doc = fitz.open(path)
        try:
            pages = extract_pdf_pages_with_ocr(doc)
        finally:
            doc.close()

        assert len(pages) == 1
        assert pages[0]["ocr"] is True
        assert pages[0]["text"] == "OCR 提取出来的扫描页文字内容"

    def test_ocr_failure_falls_back_to_text_layer(self, tmp_path, monkeypatch):
        path = _scanned_pdf(tmp_path / "scan.pdf")

        def broken_ocr(page):
            raise RuntimeError("engine crashed")

        monkeypatch.setattr(ocr_router, "OCR_ENABLED", True)
        monkeypatch.setattr(ocr_router, "ocr_page", broken_ocr)

        doc = fitz.open(path)
        try:
            pages = extract_pdf_pages_with_ocr(doc)
        finally:
            doc.close()

        # OCR 失败退回文字层提取，扫描页文本为空但不抛错
        assert len(pages) == 1
        assert pages[0]["ocr"] is False
        assert pages[0]["text"] == ""

    def test_ocr_disabled_keeps_old_behavior(self, tmp_path, monkeypatch):
        path = _scanned_pdf(tmp_path / "scan.pdf")

        monkeypatch.setattr(ocr_router, "OCR_ENABLED", False)

        # 禁用时不应触发 OCR
        def must_not_call(page):
            raise AssertionError("OCR 不应被调用")

        monkeypatch.setattr(ocr_router, "ocr_page", must_not_call)

        doc = fitz.open(path)
        try:
            pages = extract_pdf_pages_with_ocr(doc)
        finally:
            doc.close()

        assert pages[0]["ocr"] is False

    def test_mixed_pdf_only_scanned_pages_ocr(self, tmp_path, monkeypatch):
        """混合文档：文本页走文字层，扫描页走 OCR。"""
        doc = fitz.open()
        # 第 1 页：正常文本页
        page1 = doc.new_page()
        page1.insert_text(
            (72, 100),
            "本章系统介绍检索增强生成技术在课程资料问答系统中的具体应用方式与工程实践要点。",
            fontsize=11,
            fontname=CJK_FONT,
        )
        # 第 2 页：扫描页（无文字层）
        doc.new_page()

        path = tmp_path / "mixed.pdf"
        doc.save(str(path))
        doc.close()

        monkeypatch.setattr(ocr_router, "OCR_ENABLED", True)
        monkeypatch.setattr(ocr_router, "ocr_page", lambda page: "扫描页 OCR 文本")

        doc = fitz.open(path)
        try:
            pages = extract_pdf_pages_with_ocr(doc)
        finally:
            doc.close()

        assert [page["ocr"] for page in pages] == [False, True]
        assert "检索增强生成" in pages[0]["text"]
        assert pages[1]["text"] == "扫描页 OCR 文本"


class TestPdfLoaderIntegration:
    def test_text_backend_routes_scanned_pages(self, tmp_path, monkeypatch):
        """text 后端：检测到扫描页时自动走 OCR 提取。"""
        path = _scanned_pdf(tmp_path / "scan.pdf")

        monkeypatch.setattr(ocr_router, "OCR_ENABLED", True)
        monkeypatch.setattr(
            ocr_router,
            "ocr_page",
            lambda page: "OCR 提取的扫描页内容",
        )

        pages = pdf_loader.load_pdf(path, backend="text")

        assert pages[0]["text"] == "OCR 提取的扫描页内容"

    def test_markdown_backend_backfills_scanned_pages(self, tmp_path, monkeypatch):
        """markdown 后端：扫描页空文本被 OCR 补齐。"""
        path = _scanned_pdf(tmp_path / "scan.pdf")

        monkeypatch.setattr(ocr_router, "OCR_ENABLED", True)
        monkeypatch.setattr(ocr_router, "ocr_page", lambda page: "OCR 补齐的扫描页内容")

        pages = pdf_loader.load_pdf(path, backend="markdown")

        assert any("OCR 补齐的扫描页内容" in page["text"] for page in pages)


class TestEngineManagement:
    def test_unsupported_backend_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ocr_router, "OCR_BACKEND", "unknown")
        ocr_router.reset_ocr_engine()

        doc = fitz.open()
        doc.new_page()
        try:
            with pytest.raises(ValueError):
                ocr_page(doc[0])
        finally:
            doc.close()
