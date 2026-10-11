"""
image_extractor / vlm_captioner 单元测试。

图片提取用真实构造的 PDF（嵌入真实 PNG）；VLM 调用用 monkeypatch 伪造
Ollama 响应，不依赖本地 Ollama 服务。
"""

import base64

import fitz
import pytest
import requests

import modules.vlm_captioner as vlm_captioner
from modules.image_extractor import build_image_chunk, extract_document_images
from modules.vlm_captioner import caption_image, caption_images


def _png_bytes(width=300, height=200, color=(30, 100, 200)):
    """生成一张纯色 PNG（通过 pixmap 导出）。"""
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height))
    pix.set_rect(pix.irect, color)
    data = pix.tobytes("png")
    pix = None
    doc.close()
    return data


def _pdf_with_image(path, image_data, pages=1):
    """生成嵌入了图片的 PDF。"""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(fitz.Rect(72, 72, 372, 272), stream=image_data)
    doc.save(str(path))
    doc.close()
    return str(path)


class TestExtractDocumentImages:
    def test_extracts_embedded_image(self, tmp_path):
        pdf = _pdf_with_image(tmp_path / "fig.pdf", _png_bytes())

        images = extract_document_images(pdf, "hash-1", store_root=str(tmp_path / "imgs"))

        assert len(images) == 1
        assert images[0]["page"] == 1
        assert images[0]["width"] == 300
        assert images[0]["height"] == 200
        assert images[0]["image_path"].endswith(".png")
        # 文件真实写盘
        with open(images[0]["image_path"], "rb") as f:
            assert f.read(4) == b"\x89PNG"

    def test_same_xref_extracted_once(self, tmp_path):
        """同一 xref 多页引用只提取一次。"""
        pdf = _pdf_with_image(tmp_path / "multi.pdf", _png_bytes(), pages=3)

        images = extract_document_images(pdf, "hash-1", store_root=str(tmp_path / "imgs"))

        assert len(images) == 1

    def test_small_image_filtered(self, tmp_path):
        pdf = _pdf_with_image(tmp_path / "icon.pdf", _png_bytes(width=32, height=32))

        images = extract_document_images(pdf, "hash-1", store_root=str(tmp_path / "imgs"))

        assert images == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_document_images(str(tmp_path / "none.pdf"), "hash-1")

    def test_reuse_existing_extraction(self, tmp_path):
        """重复提取同一文档时复用已保存的图片文件。"""
        pdf = _pdf_with_image(tmp_path / "fig.pdf", _png_bytes())
        root = tmp_path / "imgs"

        first = extract_document_images(pdf, "hash-1", store_root=str(root))
        second = extract_document_images(pdf, "hash-1", store_root=str(root))

        assert first == second


class TestBuildImageChunk:
    def test_chunk_structure(self):
        image = {
            "page": 3,
            "image_path": "/abs/path/p0003_x17.png",
            "width": 400,
            "height": 300,
            "xref": 17,
        }

        chunk = build_image_chunk(image, "展示检索流程的示意图", "lecture.pdf")

        assert chunk["page"] == 3
        assert chunk["source"] == "lecture.pdf"
        assert chunk["heading"] == "课件图片"
        assert chunk["image_path"] == "/abs/path/p0003_x17.png"
        # 负数 chunk_id 避免与文本 chunk 冲突
        assert chunk["chunk_id"] == -17
        assert "检索流程" in chunk["text"]
        assert "第3页" in chunk["text"]


class TestVlmCaptioner:
    def test_caption_image_calls_ollama_chat(self, tmp_path, monkeypatch):
        img = tmp_path / "fig.png"
        img.write_bytes(_png_bytes())

        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json

            class FakeResponse:
                status_code = 200

                def json(self):
                    return {"message": {"content": "展示系统架构的示意图"}}

            return FakeResponse()

        monkeypatch.setattr(vlm_captioner.requests, "post", fake_post)

        caption = caption_image(str(img))

        assert caption == "展示系统架构的示意图"
        assert captured["url"].endswith("/api/chat")
        # 图片以 base64 编码放入 messages
        messages = captured["json"]["messages"]
        assert messages[0]["images"]
        base64.b64decode(messages[0]["images"][0])

    def test_caption_image_error_on_bad_status(self, tmp_path, monkeypatch):
        img = tmp_path / "fig.png"
        img.write_bytes(_png_bytes())

        class BadResponse:
            status_code = 500
            text = "server error"

            def json(self):
                return {}

        monkeypatch.setattr(
            vlm_captioner.requests, "post", lambda *a, **kw: BadResponse()
        )

        with pytest.raises(RuntimeError):
            caption_image(str(img))

    def test_caption_images_skips_failures(self, tmp_path, monkeypatch):
        images = [
            {"page": 1, "image_path": str(tmp_path / "a.png"), "width": 10, "height": 10, "xref": 1},
            {"page": 2, "image_path": str(tmp_path / "b.png"), "width": 10, "height": 10, "xref": 2},
        ]
        (tmp_path / "a.png").write_bytes(_png_bytes())
        (tmp_path / "b.png").write_bytes(_png_bytes())

        call_count = {"n": 0}

        def flaky_caption(image_path, model=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "第一张图的描述"
            raise RuntimeError("boom")

        monkeypatch.setattr(vlm_captioner, "caption_image", flaky_caption)

        results = caption_images(images, "lecture.pdf")

        # 第一张成功，第二张失败被跳过
        assert len(results) == 1
        assert results[0]["caption"] == "第一张图的描述"

    def test_caption_images_stops_when_ollama_unreachable(self, tmp_path, monkeypatch):
        img = tmp_path / "a.png"
        img.write_bytes(_png_bytes())

        images = [
            {"page": 1, "image_path": str(img), "width": 10, "height": 10, "xref": 1},
            {"page": 2, "image_path": str(img), "width": 10, "height": 10, "xref": 2},
        ]

        def unreachable(image_path, model=None):
            raise requests.ConnectionError("ollama down")

        monkeypatch.setattr(vlm_captioner, "caption_image", unreachable)

        # 服务不可达时立即停止，不再逐张重试
        results = caption_images(images, "lecture.pdf")

        assert results == []
