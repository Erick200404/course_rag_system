from typing import List, Dict


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

        # 文本长度
        text_length = len(text)

        # 从第0个字符开始切
        start = 0

        # 只要 start 没超过文本长度，就继续切
        while start < text_length:

            # 当前 chunk 的结束位置
            end = start + chunk_size

            # 取出当前 chunk 文本
            chunk_text = text[start:end].strip()

            # 如果 chunk 不为空
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,   # 当前 chunk 编号
                        "text": chunk_text,     # chunk 内容
                        "page": page,           # 页码
                        "source": source        # 来源文件
                    }
                )

                # chunk编号+1
                chunk_id += 1

            # 下一块开始的位置
            # 减去 overlap 是为了让两个 chunk 有重叠
            start += chunk_size - chunk_overlap

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