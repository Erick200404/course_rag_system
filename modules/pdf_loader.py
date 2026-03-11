from pathlib import Path
import fitz


def load_pdf(pdf_path: str) -> list[dict]:
    """
    使用PyMuPDF（fitz）库读取 PDF，按页提取文本。注意，仅实现了文本pdf的读取，无法读取图片pdf。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        一个列表，每个元素都是一页的数据：
        {
            "page": 页码,
            "text": 页面文本,
            "source": 文件名
        }
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(f"文件不是 PDF 格式: {pdf_path}")

    doc = fitz.open(pdf_file)
    pages = []

    try:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()

            # 即使这一页文字很少，也先保留；后面可以再决定是否过滤空页
            pages.append(
                {
                    "page": i + 1,
                    "text": text,
                    "source": pdf_file.name,
                }
            )
    finally:
        doc.close()

    return pages


if __name__ == "__main__":
    test_pdf_path = "../data/test.pdf"

    try:
        result = load_pdf(test_pdf_path)
        print(f"成功读取 PDF，共 {len(result)} 页\n")

        for item in result[:3]:
            print(f"页码: {item['page']}")
            print(f"来源: {item['source']}")
            print(f"内容预览: {item['text'][:200]}")
            print("-" * 50)

    except Exception as e:
        print(f"运行出错: {e}")