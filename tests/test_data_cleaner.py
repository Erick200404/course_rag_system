"""
data_cleaner 单元测试：噪声页识别与页内清理。
"""

from modules.data_cleaner import is_noise_page, clean_page_text, filter_pages


class TestIsNoisePage:
    def test_empty_text_is_noise(self):
        assert is_noise_page("") is True
        assert is_noise_page("   \n  ") is True

    def test_short_text_is_noise(self):
        assert is_noise_page("目录") is True

    def test_toc_keyword_is_noise(self):
        text = "目录\n第一章 概述\n第二章 方法\n这是很长的目录页面内容补充"
        assert is_noise_page(text) is True

    def test_dot_leader_page_is_noise(self):
        text = "第一章 概述........ 1\n第二章 方法........ 12\n第三章 实验........ 30\n附录........ 58"
        assert is_noise_page(text) is True

    def test_pure_page_number_is_noise(self):
        assert is_noise_page("第 7 页") is True
        assert is_noise_page("page 7") is True

    def test_normal_content_is_not_noise(self):
        text = (
            "本章介绍检索增强生成的基本原理，包括文档解析、向量化和检索重排等核心步骤，"
            "并讨论它们在课程资料问答场景中的具体应用与工程实践。"
        )
        assert is_noise_page(text) is False


class TestCleanPageText:
    def test_removes_standalone_page_numbers(self):
        text = "重要正文内容第一行\n7\n- 8 -\n第 9 页\nPage 10"
        cleaned = clean_page_text(text)
        assert "重要正文内容第一行" in cleaned
        assert "7" not in cleaned.split("\n")
        assert "- 8 -" not in cleaned
        assert "第 9 页" not in cleaned
        assert "Page 10" not in cleaned.lower()

    def test_keeps_normal_lines(self):
        text = "这是正文内容\n这是第二行内容"
        cleaned = clean_page_text(text)
        assert "这是正文内容" in cleaned
        assert "这是第二行内容" in cleaned


class TestFilterPages:
    def test_filters_noise_and_keeps_content(self):
        pages = [
            {"page": 1, "text": "目录", "source": "a.pdf"},
            {"page": 2, "text": "", "source": "a.pdf"},
            {
                "page": 3,
                "text": "本章介绍检索增强生成的基本原理与工程实践，内容足够长不会被过滤掉。",
                "source": "a.pdf",
            },
        ]

        filtered = filter_pages(pages)

        assert len(filtered) == 1
        assert filtered[0]["page"] == 3
        assert filtered[0]["source"] == "a.pdf"
