"""
VLM 图片描述生成。

支持智增增豆包 API 与本地 Ollama，把课件图片转成可检索的中文描述。
服务不可用时告警并停止该批图片调用，文本仍可入库。

依赖懒加载：import 本模块不会连接 Ollama。
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from config import (
    VLM_API_MODEL, VLM_BACKEND, VLM_BASE_URL, VLM_MAX_TOKENS,
    VLM_MODEL, VLM_TIMEOUT,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# 描述生成的提示词：约束输出为简洁的中文知识点描述，便于检索
CAPTION_PROMPT = (
    "你是课件分析助手。请用一两句中文客观描述这张图片的内容："
    "如果是图表请说明图表类型与展示的核心信息（如趋势、对比、流程步骤）；"
    "如果是公式或代码截图请转写其中的关键内容；"
    "如果是示意图请说明其表达的结构或概念。只输出描述本身。"
)


class VlmServiceError(RuntimeError):
    """配置、鉴权或服务故障，不应对同批每张图片重复请求。"""


def caption_fingerprint() -> str:
    """模型或提示词改变时更新索引，密钥不参与持久化。"""
    model = VLM_API_MODEL if VLM_BACKEND == "api" else VLM_MODEL
    endpoint = ((os.getenv("ZHI_BASE_URL") or "https://api.zhizengzeng.com/v1").rstrip("/")
                if VLM_BACKEND == "api" else VLM_BASE_URL)
    return hashlib.sha256(
        f"{VLM_BACKEND}:{model}:{endpoint}:{VLM_MAX_TOKENS}:{CAPTION_PROMPT}".encode()
    ).hexdigest()[:16]


def _encode_image_base64(image_path: str) -> str:
    """
    把图片文件编码为 base64 字符串。
    """
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def caption_image(image_path: str, model: str | None = None) -> str:
    """
    按 VLM_BACKEND 为单张图片生成中文描述。

    参数:
        image_path: 图片文件路径
        model: 可选模型覆盖，默认使用所选后端的模型配置

    返回:
        图片的中文描述文本；调用失败时抛出异常（由上层决定降级策略）
    """
    if VLM_BACKEND == "api":
        return _caption_api(image_path, model or VLM_API_MODEL)
    if VLM_BACKEND != "ollama":
        raise VlmServiceError(f"不支持的 VLM_BACKEND: {VLM_BACKEND}")

    image_b64 = _encode_image_base64(image_path)

    payload = {
        "model": model or VLM_MODEL,
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


def _caption_api(image_path: str, model: str) -> str:
    api_key = os.getenv("ZHI_API_KEY")
    if not api_key:
        raise VlmServiceError("未配置 ZHI_API_KEY，无法使用 API 图片理解")
    base_url = (os.getenv("ZHI_BASE_URL") or "https://api.zhizengzeng.com/v1").rstrip("/")
    mime = mimetypes.guess_type(image_path)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        # PDF 可能提取出 TIFF 等格式，转换为 PNG 后发送。
        import fitz
        pix = fitz.Pixmap(image_path)
        if pix.colorspace is None or pix.colorspace.n != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        mime = "image/png"
    else:
        image_b64 = _encode_image_base64(image_path)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": CAPTION_PROMPT},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime};base64,{image_b64}"
            }},
        ]}],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": VLM_MAX_TOKENS,
    }
    if model.startswith("doubao-"):
        payload["thinking"] = {"type": "disabled"}
    response = requests.post(
        f"{base_url}/chat/completions", json=payload,
        headers={"Authorization": f"Bearer {api_key}"}, timeout=VLM_TIMEOUT,
    )
    if response.status_code != 200:
        # 不回显响应正文，避免网关回显请求里的密钥或图片。
        hint = {
            401: "请检查 ZHI_API_KEY", 403: "请检查密钥权限或余额",
            404: "请检查 ZHI_BASE_URL 和 VLM_API_MODEL", 429: "额度或请求频率受限",
        }.get(response.status_code, "请检查模型服务状态及图片请求参数")
        raise VlmServiceError(f"API VLM 调用失败（模型 {model}），HTTP {response.status_code}：{hint}")
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise VlmServiceError("API VLM 返回缺少 choices，未生成图片描述")
    if choices[0].get("finish_reason") == "length":
        raise VlmServiceError("API VLM 描述被截断，请调高 VLM_MAX_TOKENS")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise VlmServiceError("API VLM 返回空图片描述")
    return content.strip()


def caption_images(
    images: list[dict],
    source: str,
    model: str | None = None,
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
        except VlmServiceError as e:
            print(f"[警告] {source} 图片理解服务失败，剩余图片跳过：{e}")
            break
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
