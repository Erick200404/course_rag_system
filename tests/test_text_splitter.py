"""
text_splitter 单元测试：markdown 标题切分与 text 后端兜底切分。
"""

import pytest

from modules.text_splitter import split_text, _extract_sections, _split_long_text


def _page(text, page=1, source="test.pdf"):
    return {"page": page, "text": text, "source": source}


class TestExtractSections:
    def test_heading_path(self):
        md = (
            "# 第一章 概述\n\n概述正文。\n\n"
            "## 1.1 背景\n\n背景正文。\n\n"
            "## 1.2 目标\n\n目标正文。"
        )

        sections = _extract_sections(md)

        assert len(sections) == 3
        assert sections[0] == ("第一章 概述", "概述正文。")
        assert sections[1] == ("第一章 概述 > 1.1 背景", "背景正文。")
        assert sections[2] == ("第一章 概述 > 1.2 目标", "目标正文。")

    def test_no_heading_text(self):
        sections = _extract_sections("普通正文段落，没有标题。")
        assert sections == [("", "普通正文段落，没有标题。")]


class TestMarkdownBackend:
    def test_chunks_carry_heading(self):
        md = (
            "# 第一章 概述\n\n这是概述内容，介绍整体框架。\n\n"
            "## 1.1 背景\n\n这是背景内容，说明项目动机与现状。"
        )

        chunks = split_text([_page(md)], backend="markdown")

        assert len(chunks) == 2
        assert chunks[0]["heading"] == "第一章 概述"
        assert chunks[1]["heading"] == "第一章 概述 > 1.1 背景"

        # 标题增强：chunk 文本以章节路径开头
        assert chunks[0]["text"].startswith("第一章 概述")
        assert "这是概述内容" in chunks[0]["text"]

    def test_long_section_falls_back_and_keeps_heading(self):
        # 构造一个超过 chunk_size 的区段
        body = "。".join([f"这是第{i}句用于测试长文本切分行为" for i in range(200)])
        md = f"# 第二章 方法\n\n{body}。"

        chunks = split_text([_page(md)], chunk_size=200, chunk_overlap=50, backend="markdown")

        assert len(chunks) > 1
        # 每个切分出的 chunk 都保留同一章节路径
        assert all(c["heading"] == "第二章 方法" for c in chunks)

    def test_plain_text_without_heading_still_splits(self):
        text = "这一页是没有标题结构的纯正文内容，但足够长需要被切分处理。" * 10

        chunks = split_text([_page(text)], chunk_size=100, chunk_overlap=20, backend="markdown")

        assert len(chunks) >= 1
        # 无标题时不应产生 heading 字段
        assert all("heading" not in c for c in chunks)

    def test_empty_page_skipped(self):
        chunks = split_text([_page("   ")], backend="markdown")
        assert chunks == []


class TestTextBackend:
    def test_char_backend_basic(self):
        text = "第一段内容。\n\n第二段内容。"

        chunks = split_text([_page(text)], backend="text")

        assert len(chunks) == 2
        assert chunks[0]["text"] == "第一段内容。"
        assert "heading" not in chunks[0]

    def test_chunk_ids_increment(self):
        pages = [_page("第一页内容。"), _page("第二页内容。", page=2)]

        chunks = split_text(pages, backend="text")

        assert [c["chunk_id"] for c in chunks] == [0, 1]
        assert [c["page"] for c in chunks] == [1, 2]


class TestValidation:
    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError):
            split_text([_page("x")], chunk_size=100, chunk_overlap=100)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            split_text([_page("x")], backend="unknown")


class TestSplitLongText:
    def test_char_level_fallback(self):
        # 单句超长，触发字符级切分
        sentence = "字" * 500

        pieces = _split_long_text(sentence, chunk_size=100, chunk_overlap=20)

        assert len(pieces) > 1
        assert all(len(p) <= 100 for p in pieces)

    def test_overlap_present(self):
        sentence = "字" * 500

        pieces = _split_long_text(sentence, chunk_size=100, chunk_overlap=20)

        # 相邻两块之间应存在重叠
        if len(pieces) >= 2:
            tail = pieces[0][-20:]
            assert pieces[1].startswith(tail)
