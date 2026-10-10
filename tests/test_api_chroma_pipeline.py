"""
Chroma 流水线集成测试。

使用真实的 chromadb（每个用例独立临时目录）验证 api.main._build_chroma_pipeline
的整体行为，embedding 用确定性的词袋向量伪造，避免测试依赖模型下载。
"""

import math

import fitz
import pytest

import api.main as api_main
import modules.doc_registry as doc_registry
import modules.hybrid_retriever as hybrid_retriever

VOCAB = ["检索", "增强", "生成", "向量", "数据库", "索引", "课程", "模型"]


def _fake_vector(text: str) -> list[float]:
    """基于关键词词频的确定性向量，保证相似文本检索距离更近。"""
    vec = [float(text.count(word)) for word in VOCAB]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _fake_embed_chunks(chunks):
    return [
        {**chunk, "embedding": _fake_vector(str(chunk["text"]))}
        for chunk in chunks
    ]


def _fake_query_embedding(query: str) -> list[float]:
    return _fake_vector(query)


# 默认 Helvetica 不支持中文，中文文本会被渲染成乱码；
# 使用 PyMuPDF 内置的 CJK 字体，保证生成的 PDF 可被正常抽取文本。
CJK_FONT = "china-s"


def _make_pdf(path, lines):
    """生成一个测试 PDF；lines 为空时产生一页无文本的空白页。"""
    doc = fitz.open()
    page = doc.new_page()

    y = 100
    for text, fontsize in lines:
        page.insert_text((72, y), text, fontsize=fontsize, fontname=CJK_FONT)
        y += fontsize + 24

    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def chroma_env(tmp_path, monkeypatch):
    """为每个用例准备独立的 Chroma 目录、注册表与伪造 embedding。"""
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")

    monkeypatch.setattr(api_main, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(api_main, "_chroma_store", None)
    monkeypatch.setattr(doc_registry, "REGISTRY_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setattr(api_main, "embed_chunks", _fake_embed_chunks)
    monkeypatch.setattr(hybrid_retriever, "get_query_embedding", _fake_query_embedding)

    api_main._retriever_cache.clear()

    return tmp_path


def test_pipeline_indexes_selected_documents(chroma_env):
    pdf_a = _make_pdf(
        chroma_env / "a.pdf",
        [
            ("第一章 检索增强生成概述", 20),
            ("本章系统介绍检索增强生成技术在课程资料问答中的应用与工程实践要点。", 11),
            ("内容同时涵盖向量数据库、索引结构以及模型调用链的设计与优化思路。", 11),
        ],
    )
    pdf_b = _make_pdf(
        chroma_env / "b.pdf",
        [
            ("第二章 向量数据库", 20),
            ("向量数据库用于加速大规模向量的近似检索过程，是系统的核心组件之一。", 11),
        ],
    )

    index, metadata, selection_id, retriever = api_main._build_chroma_pipeline(
        [pdf_a, pdf_b]
    )

    assert selection_id
    assert {chunk["source"] for chunk in metadata} == {"a.pdf", "b.pdf"}
    assert all(chunk["text"] for chunk in metadata)

    # 选择集视图应绑定到当前知识库，混合检索能召回内容
    assert getattr(index, "is_chroma_view", False) is True
    results = retriever.retrieve("检索增强生成")
    assert results


def test_unchanged_file_skips_reembedding(chroma_env, monkeypatch):
    counted = []

    def counting_embed(chunks):
        counted.extend(chunks)
        return _fake_embed_chunks(chunks)

    monkeypatch.setattr(api_main, "embed_chunks", counting_embed)

    pdf_a = _make_pdf(
        chroma_env / "a.pdf",
        [
            ("第一章 检索增强生成", 20),
            ("本章介绍检索增强生成技术在课程资料问答系统中的具体应用方式。", 11),
        ],
    )

    api_main._build_chroma_pipeline([pdf_a])
    after_first = len(counted)

    api_main._build_chroma_pipeline([pdf_a])

    assert after_first > 0
    assert len(counted) == after_first


def test_changed_file_replaces_stale_chunks(chroma_env):
    pdf_path = chroma_env / "a.pdf"
    _make_pdf(
        pdf_path,
        [
            ("第一章 检索增强生成", 20),
            ("本章介绍检索增强生成技术在课程资料问答系统中的具体应用方式。", 11),
        ],
    )

    api_main._build_chroma_pipeline([str(pdf_path)])

    store = api_main._get_chroma_store()
    old_hash = doc_registry.get_document("a.pdf")["sha256"]

    # 用完全不同的内容覆盖同名文件
    _make_pdf(
        pdf_path,
        [
            ("附录 实验环境说明", 20),
            ("这一节记录实验环境、依赖版本以及本地推理服务的启动步骤与注意事项。", 11),
        ],
    )

    _, metadata, _, _ = api_main._build_chroma_pipeline([str(pdf_path)])
    new_hash = doc_registry.get_document("a.pdf")["sha256"]

    assert new_hash != old_hash
    assert store.get_documents([new_hash])
    # 旧版本向量必须被清理，否则检索会命中已被替换的内容
    assert store.get_documents([old_hash]) == []
    assert all(chunk["doc_hash"] == new_hash for chunk in metadata)


def test_selection_is_isolated(chroma_env):
    pdf_a = _make_pdf(
        chroma_env / "a.pdf",
        [
            ("第一章 检索增强生成", 20),
            ("本章介绍检索增强生成技术在课程资料问答系统中的具体应用方式。", 11),
        ],
    )
    pdf_b = _make_pdf(
        chroma_env / "b.pdf",
        [
            ("第二章 向量数据库", 20),
            ("向量数据库用于加速大规模向量的近似检索过程，是系统的核心组件之一。", 11),
        ],
    )

    api_main._build_chroma_pipeline([pdf_a, pdf_b])

    _, metadata, _, retriever = api_main._build_chroma_pipeline([pdf_b])

    assert {chunk["source"] for chunk in metadata} == {"b.pdf"}
    results = retriever.retrieve("向量数据库")
    assert results
    assert all(item["source"] == "b.pdf" for item in results)


def test_pdf_without_parseable_chunks_is_skipped(chroma_env):
    good_pdf = _make_pdf(
        chroma_env / "good.pdf",
        [
            ("第一章 检索增强生成", 20),
            ("本章介绍检索增强生成技术在课程资料问答系统中的具体应用方式。", 11),
        ],
    )
    # 没有任何文本的页面：解析 + 噪声过滤后不会产生 chunk
    blank_pdf = _make_pdf(chroma_env / "blank.pdf", [])

    _, metadata, _, _ = api_main._build_chroma_pipeline([good_pdf, blank_pdf])

    assert {chunk["source"] for chunk in metadata} == {"good.pdf"}
