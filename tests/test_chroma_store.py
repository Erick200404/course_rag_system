from typing import Any

from modules.chroma_store import ChromaVectorStore


class FakeCollection:
    def __init__(self, name: str, metadata: dict[str, Any]):
        self.name = name
        self.metadata = metadata
        self.records: dict[str, dict[str, Any]] = {}

    def _matches(self, metadata: dict[str, Any], where: dict[str, Any]) -> bool:
        for key, condition in where.items():
            if key == "$and":
                if not all(self._matches(metadata, item) for item in condition):
                    return False
            elif isinstance(condition, dict) and "$eq" in condition:
                if metadata.get(key) != condition["$eq"]:
                    return False
            elif isinstance(condition, dict) and "$in" in condition:
                if metadata.get(key) not in condition["$in"]:
                    return False
            elif metadata.get(key) != condition:
                return False
        return True

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for chunk_id in ids:
            self.records.pop(chunk_id, None)

        for chunk_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            self.records[chunk_id] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

    def delete(self, where: dict[str, Any]) -> None:
        for chunk_id, record in list(self.records.items()):
            if self._matches(record["metadata"], where):
                self.records.pop(chunk_id)

    def get(
        self,
        where: dict[str, Any],
        include: list[str],
        limit: int | None = None,
    ) -> dict[str, Any]:
        records = [
            record
            for record in self.records.values()
            if self._matches(record["metadata"], where)
        ][:limit]
        return {
            "ids": [str(i) for i in range(len(records))],
            "documents": [record["document"] for record in records],
            "metadatas": [record["metadata"] for record in records],
        }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any],
        include: list[str],
    ) -> dict[str, Any]:
        records = [
            record
            for record in self.records.values()
            if self._matches(record["metadata"], where)
        ][:n_results]
        return {
            "documents": [[record["document"] for record in records]],
            "metadatas": [[record["metadata"] for record in records]],
            "distances": [[0.25 for _ in records]],
        }


class FakeClient:
    def __init__(self):
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, Any],
    ) -> FakeCollection:
        if name not in self.collections:
            self.collections[name] = FakeCollection(name, metadata)
        return self.collections[name]


def _store():
    client = FakeClient()
    store = ChromaVectorStore(
        path="storage/test-chroma",
        collection_name="test_collection",
        client=client,
    )
    return store, store.collection


def _embedded_chunks():
    return [
        {
            "chunk_id": 0,
            "text": "第一章 检索增强生成",
            "page": 1,
            "source": "lecture.pdf",
            "heading": "第一章 检索增强生成",
            "embedding": [1.0, 0.0],
        },
        {
            "chunk_id": 1,
            "text": "第二章 向量数据库",
            "page": 2,
            "source": "lecture.pdf",
            "heading": "第二章 向量数据库",
            "embedding": [0.0, 1.0],
        },
    ]


def test_upsert_preserves_metadata_and_uses_cosine_collection():
    store, collection = _store()

    store.upsert_document("hash-1", "lecture.pdf", _embedded_chunks())

    assert collection.metadata == {"hnsw:space": "cosine"}
    assert len(collection.records) == 2
    assert all(
        record["metadata"]["doc_hash"] == "hash-1"
        for record in collection.records.values()
    )
    assert any(
        record["metadata"].get("heading") == "第一章 检索增强生成"
        for record in collection.records.values()
    )


def test_get_and_query_are_limited_to_selected_documents():
    store, _ = _store()
    store.upsert_document("hash-1", "lecture.pdf", _embedded_chunks())
    store.upsert_document(
        "hash-2",
        "other.pdf",
        [
            {
                "chunk_id": 0,
                "text": "另一份文档",
                "page": 1,
                "source": "other.pdf",
                "embedding": [0.5, 0.5],
            }
        ],
    )

    selected = store.get_documents(["hash-1"])
    assert len(selected) == 2
    assert {chunk["source"] for chunk in selected} == {"lecture.pdf"}

    results = store.query([1.0, 0.0], ["hash-1"], top_k=2)
    assert len(results) == 2
    assert results[0]["score"] == 0.75


def test_document_replacement_removes_stale_chunks():
    store, collection = _store()
    store.upsert_document("hash-1", "lecture.pdf", _embedded_chunks())

    replacement = [_embedded_chunks()[0]]
    store.upsert_document("hash-2", "lecture.pdf", replacement)

    assert len(collection.records) == 1
    assert list(collection.records.values())[0]["metadata"]["doc_hash"] == "hash-2"
