import numpy as np
import faiss

from modules.vector_store import build_faiss_index, search_faiss_index


def _embedded_chunk(chunk_id, embedding, heading=""):
    chunk = {
        "chunk_id": chunk_id,
        "text": f"chunk {chunk_id}",
        "page": chunk_id + 1,
        "source": "test.pdf",
        "embedding": embedding,
    }
    if heading:
        chunk["heading"] = heading
    return chunk


def test_build_index_preserves_heading_metadata():
    chunks = [_embedded_chunk(0, [3, 4], "第一章 概述")]

    index, metadata = build_faiss_index(chunks)

    assert index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert metadata[0]["heading"] == "第一章 概述"
    assert "embedding" not in metadata[0]


def test_search_uses_cosine_similarity():
    chunks = [
        _embedded_chunk(0, [3, 4], "第一章"),
        _embedded_chunk(1, [-1, 0], "第二章"),
    ]

    index, metadata = build_faiss_index(chunks)
    results = search_faiss_index(index, metadata, [6, 8], top_k=2)

    assert results[0]["chunk_id"] == 0
    assert results[0]["score"] == np.float64(1.0)
