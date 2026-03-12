from typing import List, Dict
import re   # 新增：用于识别页码、目录等模式


def is_noise_page(text: str) -> bool:
    """
    判断某一页是否为噪声页，例如目录页、过短页、无实质内容页。
    """
    if not text or not text.strip():
        return True

    cleaned_text = text.strip().lower()

    # 1. 过滤过短页面
    # 很多封面页、标题页、空白页文本非常短
    if len(cleaned_text) < 30:
        return True

    # 2. 过滤常见目录页关键词
    noise_keywords = [
        "目录",
        "contents",
        "table of contents",
    ]

    if any(keyword in cleaned_text for keyword in noise_keywords):
        return True

    # =========================
    # 新增逻辑：识别目录页结构
    # =========================
    # 目录页通常包含大量点号（......）
    if cleaned_text.count("...") > 3:
        return True

    # =========================
    # 新增逻辑：检测纯页码页
    # =========================
    # 例如只包含 “第 7 页” 或 “page 7”
    if re.fullmatch(r"(第?\s*\d+\s*页?)|(page\s*\d+)", cleaned_text):
        return True

    return False


def clean_page_text(text: str) -> str:
    """
    清理页面内部噪声，例如页码、页眉页脚。
    注意：这里只清理 text，不影响 page 元数据。
    """

    # 将文本按行拆分
    lines = text.split("\n")

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # =========================
        # 删除单独的页码行
        # =========================

        # 例如：
        # 7
        # - 7 -
        # 第 7 页
        # Page 7
        if re.fullmatch(r"-?\s*\d+\s*-?", line):
            continue

        if re.fullmatch(r"第\s*\d+\s*页", line):
            continue

        if re.fullmatch(r"page\s*\d+", line.lower()):
            continue

        # =========================
        # 删除明显的页眉页脚
        # =========================
        # 例如学校名、课程名等可以根据需要继续扩展
        if len(line) < 3:
            continue

        cleaned_lines.append(line)

    # 将清理后的文本重新拼接
    return "\n".join(cleaned_lines)


def filter_pages(pages: List[Dict]) -> List[Dict]:
    """
    过滤 PDF 解析后的噪声页。
    """

    filtered_pages = []

    for page in pages:

        text = page["text"]

        # =========================
        # 第一步：判断是否整页噪声
        # =========================
        if is_noise_page(text):
            continue

        # =========================
        # 第二步：清理页内噪声
        # =========================
        cleaned_text = clean_page_text(text)

        # 如果清理后文本过短，说明信息价值不高
        if len(cleaned_text.strip()) < 30:
            continue

        # 构造新的 page 数据（保留 page 元数据）
        filtered_pages.append(
            {
                "page": page["page"],       # 保留页码
                "text": cleaned_text,       # 使用清理后的文本
                "source": page["source"]    # 保留来源
            }
        )

    return filtered_pages