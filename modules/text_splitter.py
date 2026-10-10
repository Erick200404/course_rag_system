from typing import List, Dict

import re

from config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_PARSE_BACKEND

# Markdown 标题行，例如 "# 第一章 概述"、"## 1.1 背景"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _make_chunk(chunk_id: int, text: str, page: int, source: str, heading: str = "") -> Dict:
    """
    构造统一的 chunk 结构。
    """
    chunk = {
        "chunk_id": chunk_id,
        "text": text,
        "page": page,
        "source": source,
    }

    # markdown 后端会产生章节路径，用于展示和 embedding 标题增强
    if heading:
        chunk["heading"] = heading

    return chunk


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    长文本兜底切分：段落 -> 句子 -> 字符 三级策略。

    返回切分后的纯文本列表（不带元数据）。
    """
    texts = []

    # 两个及以上换行视为段落分隔
    paragraphs = re.split(r"\n\s*\n+", text)

    for para in paragraphs:
        para = para.strip()

        if not para:
            continue

        # 段落本身够短，直接作为一段
        if len(para) <= chunk_size:
            texts.append(para)
            continue

        # 段落太长 -> 按句子切分（中英文句尾符号）
        sentences = re.split(r"(?<=[。！？!?\.])", para)

        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # 单句超过 chunk_size -> 字符级兜底
            if len(sentence) > chunk_size:
                if current_chunk:
                    texts.append(current_chunk.strip())

                start = 0
                while start < len(sentence):
                    sub_text = sentence[start:start + chunk_size].strip()
                    if sub_text:
                        texts.append(sub_text)
                    start += chunk_size - chunk_overlap

                current_chunk = ""
                continue

            # 当前 chunk 加上新句子不会超长
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence

            else:
                # 当前 chunk 已满，先保存
                if current_chunk:
                    texts.append(current_chunk.strip())

                # overlap 处理：保留上一块末尾部分
                if chunk_overlap > 0 and current_chunk:
                    current_chunk = current_chunk[-chunk_overlap:] + sentence
                else:
                    current_chunk = sentence

        # 段落内最后剩余的内容
        if current_chunk.strip():
            texts.append(current_chunk.strip())

    return texts


def _split_char_pages(pages: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """
    text 后端：整页走三级兜底切分（原有行为）。
    """
    chunks = []
    chunk_id = 0

    for page_data in pages:
        text = page_data["text"]
        page = page_data["page"]
        source = page_data["source"]

        if not text.strip():
            continue

        for piece in _split_long_text(text, chunk_size, chunk_overlap):
            chunks.append(_make_chunk(chunk_id, piece, page, source))
            chunk_id += 1

    return chunks


def _extract_sections(markdown_text: str) -> List[tuple]:
    """
    将一页 Markdown 文本按标题层级切出区段。

    返回:
        [(章节路径, 区段正文), ...]
        章节路径形如 "第一章 概述 > 1.1 背景"
    """
    sections = []

    # 标题栈，例如 [(1, "第一章"), (2, "1.1 背景")]
    stack = []
    buf = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            heading = " > ".join(title for _, title in stack)
            sections.append((heading, body))

    for line in markdown_text.split("\n"):
        match = HEADING_RE.match(line)

        if match:
            # 先把上一个标题下的内容收尾
            flush()
            buf = []

            level = len(match.group(1))
            title = match.group(2).strip()

            # 弹出同级及更浅层级的标题，维护标题路径栈
            while stack and stack[-1][0] >= level:
                stack.pop()

            stack.append((level, title))
        else:
            buf.append(line)

    # 收尾最后一个区段
    flush()

    return sections


def _split_markdown_pages(pages: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """
    markdown 后端：标题层级感知切分。

    逻辑：
    1. 按标题把页面切为区段，并记录章节路径（标题栈）
    2. 区段不超长 -> 整段作为一个 chunk
    3. 区段超长 -> 回退到 段落/句子/字符 三级兜底切分
    4. chunk 文本前面拼上章节路径（标题增强，有利于 embedding 检索）
    """
    chunks = []
    chunk_id = 0

    for page_data in pages:
        text = page_data["text"]
        page = page_data["page"]
        source = page_data["source"]

        if not text.strip():
            continue

        for heading, body in _extract_sections(text):
            if not body:
                continue

            # 区段超长时回退到兜底切分
            if len(body) > chunk_size:
                pieces = _split_long_text(body, chunk_size, chunk_overlap)
            else:
                pieces = [body]

            for piece in pieces:
                # 标题增强：把章节路径拼进 chunk 文本
                chunk_text = f"{heading}\n{piece}" if heading else piece

                chunks.append(
                    _make_chunk(chunk_id, chunk_text, page, source, heading)
                )
                chunk_id += 1

    return chunks


def split_text(
        pages: List[Dict],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        backend: str | None = None,
) -> List[Dict]:
    """
    将按页提取的 PDF 文本切分为多个 chunk。

    参数:
        pages: pdf_loader 返回的数据
               结构类似:
               [
                   {"page":1, "text":"...", "source":"test.pdf"},
                   {"page":2, "text":"...", "source":"test.pdf"}
               ]

        chunk_size: 每个 chunk 的最大字符数

        chunk_overlap: 相邻 chunk 之间重叠的字符数

        backend: 切分后端，"markdown"（标题感知）或 "text"（三级兜底）；
                 不传时使用 config.PDF_PARSE_BACKEND

    返回:
        chunks 列表，例如:
        [
            {
                "chunk_id": 0,
                "text": "...",
                "page": 1,
                "source": "test.pdf",
                "heading": "第一章 概述"   # 仅 markdown 后端存在
            }
        ]
    """
    # overlap 必须小于 chunk_size，否则会死循环
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    backend = backend or PDF_PARSE_BACKEND

    if backend == "markdown":
        return _split_markdown_pages(pages, chunk_size, chunk_overlap)

    if backend == "text":
        return _split_char_pages(pages, chunk_size, chunk_overlap)

    raise ValueError(f"不支持的切分 backend: {backend}")


# 下面是一个测试代码
if __name__ == "__main__":

    # 从 pdf_loader 导入函数
    from modules.pdf_loader import load_pdf

    # 读取测试 PDF
    pages = load_pdf("data/test.pdf")

    # 进行 chunk 切分
    chunks = split_text(pages)

    # 打印生成了多少 chunk
    print(f"共生成 {len(chunks)} 个 chunks\n")

    # 只展示前5个 chunk
    for chunk in chunks[:5]:
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"page: {chunk['page']}")
        print(f"source: {chunk['source']}")
        print(f"heading: {chunk.get('heading', '(无)')}")
        print(f"text preview: {chunk['text'][:200]}")
        print("-" * 50)
