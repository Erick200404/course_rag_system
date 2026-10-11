from pathlib import Path

import fitz

from config import PDF_PARSE_BACKEND
from modules.ocr_router import extract_pdf_pages_with_ocr, is_scanned_page


def _validate_pdf(pdf_path: str) -> Path:
    """
    校验 PDF 路径合法性的公共入口。
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(f"文件不是 PDF 格式: {pdf_path}")

    return pdf_file


def _load_pdf_text(pdf_file: Path) -> list[dict]:
    """
    后端一：使用 PyMuPDF（fitz）逐页提取纯文本。

    扫描页（无文字层）会自动路由到 OCR（见 modules/ocr_router.py）；
    OCR 未启用时退回文字层提取，扫描页文本为空由 data_cleaner 过滤。

    Returns:
        一个列表，每个元素都是一页的数据：
        {
            "page": 页码,
            "text": 页面文本,
            "source": 文件名,
            "ocr": 是否经过 OCR（可选字段）
        }
    """
    doc = fitz.open(pdf_file)

    try:
        # 有扫描页且 OCR 可用时走 OCR 提取
        has_scan = any(is_scanned_page(page) for page in doc)
        if has_scan:
            return extract_pdf_pages_with_ocr(doc)

        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()

            # 即使这一页文字很少，也先保留；由 data_cleaner 决定是否过滤
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


def _load_pdf_markdown(pdf_file: Path) -> list[dict]:
    """
    后端二：使用 pymupdf4llm 将 PDF 转为 Markdown。

    优势（相对纯文本提取）：
    - 基于字体大小识别标题层级，输出 #/##/### 结构
    - 表格转换为 Markdown 表格
    - 按版面块推断阅读顺序（对双栏课件友好）

    Returns:
        与 _load_pdf_text 相同结构的数据，text 为该页的 Markdown 文本。
        扫描页（Markdown 为空且无文字层）会补充 OCR 文本。
    """
    try:
        import pymupdf4llm
    except ImportError as e:
        raise ImportError(
            "markdown 解析后端需要先安装 pymupdf4llm：pip install pymupdf4llm"
        ) from e

    # page_chunks=True 时按页返回，每页一个 dict：
    # {"metadata": {"page": 从 0 开始的页码, ...}, "text": 该页 Markdown}
    md_pages = pymupdf4llm.to_markdown(str(pdf_file), page_chunks=True)

    pages = []
    for idx, md_page in enumerate(md_pages):
        text = (md_page.get("text") or "").strip()

        # 页码兼容处理：
        # - 新版 pymupdf4llm 返回 page_number（1 起始）
        # - 旧版返回 page（0 起始，需 +1）
        # - 都没有时回退到列表下标
        meta = md_page.get("metadata") or {}
        if meta.get("page_number") is not None:
            page_no = int(meta["page_number"])
        elif meta.get("page") is not None:
            page_no = int(meta["page"]) + 1
        else:
            page_no = idx + 1

        pages.append(
            {
                "page": page_no,
                "text": text,
                "source": pdf_file.name,
            }
        )

    # 扫描页在 pymupdf4llm 输出中是空文本，用 OCR 补齐
    pages = _backfill_scanned_pages_with_ocr(pdf_file, pages)

    return pages


def _backfill_scanned_pages_with_ocr(pdf_file: Path, pages: list[dict]) -> list[dict]:
    """
    对 markdown 后端输出中的扫描页做 OCR 补齐。

    markdown 后端无法按页判断扫描页（输出只有文本），这里重新打开
    文档按页码对齐检测，仅对扫描页执行 OCR。
    """
    from modules.ocr_router import OCR_ENABLED, ocr_page

    if not OCR_ENABLED:
        return pages

    doc = fitz.open(pdf_file)
    try:
        scanned_page_nos = [
            page.number + 1 for page in doc if is_scanned_page(page)
        ]
    finally:
        doc.close()

    if not scanned_page_nos:
        return pages

    print(f"[OCR] {pdf_file.name} 检测到 {len(scanned_page_nos)} 个扫描页，尝试 OCR 补齐")

    doc = fitz.open(pdf_file)
    try:
        for page_data in pages:
            if page_data["page"] not in scanned_page_nos:
                continue

            try:
                ocr_text = ocr_page(doc[page_data["page"] - 1])
            except (ImportError, ValueError) as e:
                print(f"[警告] OCR 不可用，第 {page_data['page']} 页保持为空：{e}")
                continue
            except Exception as e:
                print(f"[警告] 第 {page_data['page']} 页 OCR 失败：{e}")
                continue

            if ocr_text:
                page_data["text"] = ocr_text
                page_data["ocr"] = True
    finally:
        doc.close()

    return pages


def load_pdf(pdf_path: str, backend: str | None = None) -> list[dict]:
    """
    读取 PDF，按页返回文本数据。

    参数:
        pdf_path: PDF 文件路径
        backend: 解析后端，"markdown" 或 "text"；
                 不传时使用 config.PDF_PARSE_BACKEND

    返回:
        [{"page": 页码, "text": 文本, "source": 文件名}, ...]
    """
    pdf_file = _validate_pdf(pdf_path)

    backend = backend or PDF_PARSE_BACKEND

    if backend == "markdown":
        return _load_pdf_markdown(pdf_file)

    if backend == "text":
        return _load_pdf_text(pdf_file)

    raise ValueError(f"不支持的 PDF_PARSE_BACKEND: {backend}")


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
