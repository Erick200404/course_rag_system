from typing import List, Dict
import os

from dotenv import load_dotenv
from openai import OpenAI

# 从配置文件中读取 embedding 相关配置
from config import EMBEDDING_MODEL, EMBEDDING_BACKEND, HF_EMBEDDING_MODEL

# 读取 .env 文件
load_dotenv()

# ==============================
# API embedding 客户端初始化
# ==============================
# 保留 OpenAI-compatible 接口方式，
# 这样即使切换到 HuggingFace，也不会破坏原有逻辑。
client = OpenAI(
    api_key=os.getenv("ZHI_API_KEY"),
    base_url=os.getenv("ZHI_BASE_URL"),
)

# ==============================
# HuggingFace embedding 模型初始化
# ==============================
# 只有在配置为 hf 时，才加载本地模型，
# 避免在使用 API 模式时额外占用启动时间和内存。
hf_model = None
if EMBEDDING_BACKEND == "hf":
    import torch
    from sentence_transformers import SentenceTransformer

    # 自动检测 CUDA 并使用 GPU 推理
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model = SentenceTransformer(HF_EMBEDDING_MODEL, device=device)

    print(f"当前 embedding 设备: {device}")

# BGE 系列模型检索场景的查询指令前缀（官方推荐用法）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# HF 本地批量编码的批大小
HF_BATCH_SIZE = 32


def _is_bge_model() -> bool:
    """
    判断当前 HF embedding 模型是否为 BGE 系列
    （BGE 查询侧需要追加指令前缀）。
    """
    return HF_EMBEDDING_MODEL.lower().startswith("baai/bge")


def get_text_embedding(text: str) -> List[float]:
    """
    统一的单条文本 embedding 接口。

    参数:
        text: 输入文本

    返回:
        embedding 向量
    """
    # 去掉首尾空白，避免传入无效内容
    text = text.strip()

    # 如果文本为空，直接报错
    if not text:
        raise ValueError("text 不能为空")

    # ==============================
    # 使用 HuggingFace 本地 embedding
    # ==============================
    if EMBEDDING_BACKEND == "hf":
        # encode 返回 numpy 数组，这里转成 list 方便后续序列化和存储
        embedding = hf_model.encode(text)
        return embedding.tolist()

    # ==============================
    # 使用远程 API embedding
    # ==============================
    elif EMBEDDING_BACKEND == "api":
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    # 如果配置了不支持的后端，直接抛出异常
    else:
        raise ValueError(f"不支持的 EMBEDDING_BACKEND: {EMBEDDING_BACKEND}")


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    将 chunk 文本转为向量，并写回 chunk。

    参数:
        chunks: text_splitter 生成的 chunk 列表

    返回:
        带 embedding 向量的 chunk 列表
    """
    # ==============================
    # HuggingFace 后端：批量编码
    # ==============================
    # 相比逐条 encode，批量编码可以充分利用 GPU，速度提升明显
    if EMBEDDING_BACKEND == "hf":
        texts = [chunk["text"].strip() for chunk in chunks]

        vectors = hf_model.encode(
            texts,
            batch_size=HF_BATCH_SIZE,
            show_progress_bar=False,
        )

        embedded_chunks = []
        for chunk, vector in zip(chunks, vectors):
            new_chunk = chunk.copy()
            new_chunk["embedding"] = vector.tolist()
            embedded_chunks.append(new_chunk)

        return embedded_chunks

    # ==============================
    # API 后端：逐条调用
    # ==============================
    embedded_chunks = []

    for chunk in chunks:
        # 取出当前 chunk 的文本
        text = chunk["text"]

        # 通过统一接口生成 embedding
        embedding = get_text_embedding(text)

        # 复制原 chunk，避免直接修改原数据
        new_chunk = chunk.copy()

        # 加入 embedding 向量
        new_chunk["embedding"] = embedding

        # 保存到结果列表
        embedded_chunks.append(new_chunk)

    return embedded_chunks


def get_query_embedding(query: str) -> List[float]:
    """
    将用户问题转为 embedding 向量。

    BGE 系列模型在检索场景下，查询侧需要追加指令前缀，
    文档侧不需要，这里只对查询做处理。

    参数:
        query: 用户输入的问题

    返回:
        query 对应的 embedding 向量
    """
    text = query.strip()

    if EMBEDDING_BACKEND == "hf" and _is_bge_model() and text:
        text = BGE_QUERY_INSTRUCTION + text

    return get_text_embedding(text)


if __name__ == "__main__":
    from modules.pdf_loader import load_pdf
    from modules.text_splitter import split_text

    # 1. 读取 PDF
    pages = load_pdf("data/test.pdf")

    # 2. 切分 chunk
    chunks = split_text(pages)

    # 3. 调用 embedding
    embedded_chunks = embed_chunks(chunks)

    print(f"当前 embedding 后端: {EMBEDDING_BACKEND}")
    print(f"生成 {len(embedded_chunks)} 个 embedding\n")

    # 打印前两个结果，便于调试
    for item in embedded_chunks[:2]:
        print(f"chunk_id: {item['chunk_id']}")
        print(f"page: {item['page']}")
        print(f"text preview: {item['text'][:100]}")
        print(f"embedding dim: {len(item['embedding'])}")
        print("-" * 50)
