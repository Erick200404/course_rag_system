"""
PDF 图片提取。

课件 PDF 中的图表（架构图/流程图/公式截图/表格截图）承载大量知识点，
纯文本解析完全丢失这部分内容。本模块负责：

1. 提取 PDF 内嵌图片（按 xref 去重，同一图片多次引用只提取一次）
2. 按最小尺寸过滤（去掉图标、装饰线条等噪声）
3. 保存到 storage/images/<doc_hash>/，文件名含页码与 xref 便于溯源

图片的语义化（生成可检索的描述文本）由 modules/vlm_captioner.py 负责。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import fitz

from config import IMAGE_MIN_SIZE, IMAGE_STORE_PATH


def extract_document_images(
    pdf_path: str,
    doc_hash: str,
    store_root: str = IMAGE_STORE_PATH,
    min_size: int = IMAGE_MIN_SIZE,
) -> List[dict]:
    """
    提取一个 PDF 文件中的全部内嵌图片并保存到磁盘。

    参数:
        pdf_path: PDF 文件路径
        doc_hash: 文档内容 hash，作为图片存放的子目录名
        store_root: 图片根目录
        min_size: 宽或高低于该像素数的图片忽略

    返回:
        [
            {
                "page": 图片所在页码（1 起始）,
                "image_path": 保存后的图片绝对路径,
                "width": 图片宽度（像素）,
                "height": 图片高度（像素）,
                "xref": 图片在 PDF 中的对象引用,
            },
            ...
        ]
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    store_dir = Path(store_root) / doc_hash
    store_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_file)
    images: List[dict] = []
    saved_xrefs = set()

    try:
        # 同一 xref 在多页出现时只处理一次，但记录第一次出现的页码
        for page_index, page in enumerate(doc, start=1):
            for image_info in page.get_images(full=True):
                xref = image_info[0]

                if xref == 0 or xref in saved_xrefs:
                    continue

                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    # 个别损坏对象直接跳过，不影响其余图片
                    continue

                if base_image is None:
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # 尺寸过滤：图标/装饰
                if width < min_size or height < min_size:
                    continue

                image_bytes = base_image.get("image")
                ext = base_image.get("ext", "png")

                if not image_bytes:
                    continue

                filename = f"p{page_index:04d}_x{xref}.{ext}"
                image_path = store_dir / filename

                # 已存在说明该文档之前提取过，直接复用（配合 doc_hash 目录隔离）
                if not image_path.exists():
                    image_path.write_bytes(image_bytes)

                images.append(
                    {
                        "page": page_index,
                        "image_path": str(image_path.resolve()),
                        "width": width,
                        "height": height,
                        "xref": xref,
                    }
                )

                saved_xrefs.add(xref)
    finally:
        doc.close()

    return images


def build_image_chunk(image: dict, caption: str, source: str) -> dict:
    """
    把「图片 + VLM 描述」包装成与文本 chunk 兼容的结构。

    caption 作为 chunk 文本入库（可被检索），image_path 存在 metadata 里，
    前端召回该 chunk 时直接展示原图。
    """
    chunk_text = (
        f"[课件图片] 第{image['page']}页插图（{image['width']}x{image['height']}）：{caption}"
    )

    return {
        # 负数 chunk_id 避免与文本 chunk 的编号冲突（文本 chunk 从 0 递增）
        "chunk_id": -image["xref"],
        "text": chunk_text,
        "page": image["page"],
        "source": source,
        "heading": "课件图片",
        "image_path": image["image_path"],
    }
