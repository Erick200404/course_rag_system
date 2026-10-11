"""
VLM 图片描述生成。

用 Ollama 的视觉语言模型（默认 qwen2.5vl:7b）把课件图片转成中文描述，
使图表内容可被语义检索。

调用方式：Ollama /api/chat + base64 图片消息（与 ChatOllama 的多模态格式一致）。
降级策略：Ollama 未启动 / 模型未拉取 / 单图超时，都只跳过该图并告警，
不让图片处理失败中断建库流程。

依赖懒加载：import 本模块不会连接 Ollama。
"""

from __future__ import annotations

import base64
from pathlib import Path

import requests

from config import VLM_BASE_URL, VLM_MODEL, VLM_TIMEOUT

# 描述生成的提示词：约束输出为简洁的中文知识点描述，便于检索
CAPTION_PROMPT = (
    "你是课件分析助手。请用一两句中文客观描述这张图片的内容："
    "如果是图表请说明图表类型与展示的核心信息（如趋势、对比、流程步骤）；"
    "如果是公式或代码截图请转写其中的关键内容；"
    "如果是示意图请说明其表达的结构或概念。只输出描述本身。"
)


def _encode_image_base64(image_path: str) -> str:
    """
    把图片文件编码为 base64 字符串。
    """
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def caption_image(image_path: str, model: str = VLM_MODEL) -> str:
    """
    调用 Ollama 视觉模型为单张图片生成中文描述。

    参数:
        image_path: 图片文件路径
        model: Ollama 视觉模型名

    返回:
        图片的中文描述文本；调用失败时抛出异常（由上层决定降级策略）
    """
    image_b64 = _encode_image_base64(image_path)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": CAPTION_PROMPT,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    response = requests.post(
        f"{VLM_BASE_URL}/api/chat",
        json=payload,
        timeout=VLM_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama VLM 调用失败，status_code={response.status_code}，detail={response.text[:200]}"
        )

    data = response.json()

    # /api/chat 的回答在 message.content
    return (data.get("message", {}).get("content") or "").strip()


def caption_images(
    images: list[dict],
    source: str,
    model: str = VLM_MODEL,
) -> list[dict]:
    """
    批量为图片生成描述（逐张调用，单张失败跳过）。

    参数:
        images: image_extractor.extract_document_images 的返回值
        source: 来源文件名（用于日志）

    返回:
        生成成功的图片列表，每项追加 "caption" 字段
    """
    results = []

    for image in images:
        try:
            caption = caption_image(image["image_path"], model=model)
        except requests.RequestException as e:
            print(f"[警告] VLM 不可达，{source} 剩余图片全部跳过：{e}")
            break
        except Exception as e:
            print(f"[警告] 第 {image['page']} 页图片描述失败，跳过：{e}")
            continue

        if not caption:
            continue

        item = dict(image)
        item["caption"] = caption
        results.append(item)

    if results:
        print(f"[VLM] {source}：{len(results)}/{len(images)} 张图片生成描述")

    return results
