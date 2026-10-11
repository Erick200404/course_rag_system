"""
扫描页检测与 OCR 路由。

问题：课件 PDF 经常混入扫描页（拍照/扫描的图片页，没有文字层），
pymupdf4llm 对这类页面提取不出任何文本，知识库直接漏掉这部分内容。

方案：
1. 检测：按「文本密度 = 字符数 / 页面面积（平方英寸）」判定扫描页；
   无文字层的扫描页密度接近 0，正常文本页在 1~6 之间
2. 路由：检测到扫描页时，把该页渲染成图片交给 OCR 后端提取文本
3. 后端可插拔：默认 RapidOCR（ONNXRuntime，纯 pip，CPU 可跑）；
   OCR_BACKEND="none" 时退回旧行为（扫描页跳过）

注意：OCR 引擎懒加载，未启用时导入本模块不会下载模型。
"""

from __future__ import annotations

from typing import Any, List

import fitz

from config import (
    OCR_BACKEND,
    OCR_ENABLED,
    SCAN_TEXT_DENSITY_THRESHOLD,
)

# 页面渲染 DPI（OCR 识别精度与速度的折中）
RENDER_DPI = 200

# OCR 引擎单例（懒加载）
_ocr_engine: Any | None = None


def text_density(page: "fitz.Page") -> float:
    """
    计算页面文本密度：可见字符数 / 页面面积（平方英寸）。

    正常 A4 文本页约 1~6，扫描页（无文字层）接近 0。
    """
    width_inch = page.rect.width / 72.0
    height_inch = page.rect.height / 72.0

    area = max(width_inch * height_inch, 1e-6)
    char_count = len(page.get_text("text").strip())

    return char_count / area


def is_scanned_page(page: "fitz.Page", threshold: float = SCAN_TEXT_DENSITY_THRESHOLD) -> bool:
    """
    判断某一页是否为扫描页（无文字层或文字极少）。
    """
    return text_density(page) < threshold


def _get_ocr_engine() -> Any:
    """
    懒加载 OCR 引擎（首次调用时才初始化，避免启动即下载模型）。
    """
    global _ocr_engine

    if _ocr_engine is None:
        if OCR_BACKEND != "rapidocr":
            raise ValueError(f"不支持的 OCR_BACKEND: {OCR_BACKEND}")

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise ImportError(
                "OCR 后端需要先安装 rapidocr_onnxruntime："
                "pip install rapidocr_onnxruntime"
            ) from e

        _ocr_engine = RapidOCR()
        print("OCR 引擎已加载: RapidOCR")

    return _ocr_engine


def ocr_page(page: "fitz.Page") -> str:
    """
    把一页渲染成图片并 OCR 提取文本。

    返回:
        OCR 识别出的文本（可能为空字符串）
    """
    engine = _get_ocr_engine()

    # 渲染页面为 PNG 图片字节流
    pixmap = page.get_pixmap(dpi=RENDER_DPI)
    image_bytes = pixmap.tobytes("png")

    # RapidOCR 输入 numpy 数组或图片路径；bytes 直接支持
    result, _ = engine(image_bytes)

    if not result:
        return ""

    # result: [[box, text, score], ...] -> 按阅读顺序拼接
    lines = [item[1] for item in result if len(item) >= 2]
    return "\n".join(lines).strip()


def extract_pdf_pages_with_ocr(doc: "fitz.Document") -> List[dict]:
    """
    按 PDF 文档逐页提取文本，扫描页自动路由到 OCR。

    参数:
        doc: 已打开的 fitz.Document

    返回:
        [{"page": 页码, "text": 文本, "source": 文件名, "ocr": bool}, ...]
        ocr 字段标记该页是否经过 OCR，便于调试与展示。
    """
    source = doc.name
    pages = []
    ocr_count = 0

    for i, page in enumerate(doc):
        text = ""
        used_ocr = False

        if OCR_ENABLED and is_scanned_page(page):
            try:
                text = ocr_page(page)
                used_ocr = bool(text)
                if used_ocr:
                    ocr_count += 1
            except (ImportError, ValueError) as e:
                # OCR 引擎不可用时退回文字层提取，不让单个环境问题中断解析
                print(f"[警告] OCR 不可用，第 {i + 1} 页按普通页处理：{e}")
            except Exception as e:
                print(f"[警告] 第 {i + 1} 页 OCR 失败：{e}")

        if not text:
            text = page.get_text("text").strip()

        pages.append(
            {
                "page": i + 1,
                "text": text,
                "source": source,
                "ocr": used_ocr,
            }
        )

    if ocr_count:
        print(f"[OCR] {source}：共 {ocr_count}/{len(pages)} 页通过 OCR 提取")

    return pages


def reset_ocr_engine() -> None:
    """
    释放 OCR 引擎单例（测试用）。
    """
    global _ocr_engine
    _ocr_engine = None
