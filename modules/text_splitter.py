from typing import List, Dict
import re   # 新增：用于正则切分段落和句子


def split_text(
        pages: List[Dict],
        chunk_size: int = 500,
        chunk_overlap: int = 100
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

    返回:
        chunks 列表，例如:
        [
            {
                "chunk_id":0,
                "text":"...",
                "page":1,
                "source":"test.pdf"
            }
        ]
    """

    # overlap 必须小于 chunk_size，否则会死循环
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    # 最终所有 chunk 会放到这里
    chunks = []

    # chunk 的编号（方便调试）
    chunk_id = 0

    # 遍历每一页
    for page_data in pages:

        # 当前页文本
        text = page_data["text"]

        # 当前页码
        page = page_data["page"]

        # 来源文件
        source = page_data["source"]

        # 如果这一页没有文本，跳过
        if not text.strip():
            continue

        # =========================
        # 新增逻辑：优先按段落切分
        # =========================
        # 两个及以上换行视为段落分隔
        paragraphs = re.split(r"\n\s*\n+", text)

        # 遍历每个段落
        for para in paragraphs:

            # 去掉首尾空白
            para = para.strip()

            # 如果段落为空，跳过
            if not para:
                continue

            # 如果段落长度本身就小于 chunk_size，直接作为 chunk
            if len(para) <= chunk_size:

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": para,
                        "page": page,
                        "source": source
                    }
                )

                chunk_id += 1

                continue

            # =========================
            # 新增逻辑：段落太长 → 按句子切
            # =========================
            # 使用中文和英文句号等符号进行句子切分
            sentences = re.split(r'(?<=[。！？!?\.])', para)

            current_chunk = ""

            # 遍历句子
            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                # 如果单句就已经超过 chunk_size
                # 说明句子非常长，需要字符级兜底切分
                if len(sentence) > chunk_size:

                    # 如果当前 chunk 有内容，先保存
                    if current_chunk:

                        chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "text": current_chunk.strip(),
                                "page": page,
                                "source": source
                            }
                        )

                        chunk_id += 1
                        current_chunk = ""

                    # =========================
                    # 字符级切分（兜底方案）
                    # =========================
                    start = 0
                    text_length = len(sentence)

                    while start < text_length:

                        end = start + chunk_size

                        sub_text = sentence[start:end].strip()

                        if sub_text:
                            chunks.append(
                                {
                                    "chunk_id": chunk_id,
                                    "text": sub_text,
                                    "page": page,
                                    "source": source
                                }
                            )

                            chunk_id += 1

                        start += chunk_size - chunk_overlap

                    continue

                # 如果当前 chunk 加上新句子不会超长
                if len(current_chunk) + len(sentence) <= chunk_size:

                    current_chunk += sentence

                else:
                    # 当前 chunk 已满，先保存
                    if current_chunk:

                        chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "text": current_chunk.strip(),
                                "page": page,
                                "source": source
                            }
                        )

                        chunk_id += 1

                    # overlap 处理：保留上一块末尾部分
                    if chunk_overlap > 0 and current_chunk:
                        tail = current_chunk[-chunk_overlap:]
                        current_chunk = tail + sentence
                    else:
                        current_chunk = sentence

            # 当前段落最后剩余的 chunk
            if current_chunk.strip():

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": current_chunk.strip(),
                        "page": page,
                        "source": source
                    }
                )

                chunk_id += 1

    # 返回所有 chunk
    return chunks


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
        print(f"text preview: {chunk['text'][:200]}")
        print("-" * 50)