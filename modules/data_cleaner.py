from typing import List, Dict


def is_noise_page(text: str) -> bool:
    """
    判断某一页是否为噪声页，例如目录页、过短页、无实质内容页。
    """
    if not text or not text.strip():
        return True

    cleaned_text = text.strip().lower()

    # 1. 过滤过短页面
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

    return False


def filter_pages(pages: List[Dict]) -> List[Dict]:
    """
    过滤 PDF 解析后的噪声页。
    """
    filtered_pages = []

    for page in pages:
        text = page["text"]

        if is_noise_page(text):
            continue

        filtered_pages.append(page)

    return filtered_pages