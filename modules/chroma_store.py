from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Sequence

from config import CHROMA_PATH


EmbeddingChunk = Dict[str, Any]
QueryEmbedding = Sequence[float]


def _chunk_id(source: str, doc_hash: str, chunk: EmbeddingChunk) -> str:
    """Build a stable Chroma ID for a document chunk."""
    raw_id = f"{source}:{doc_hash}:{chunk['chunk_id']}:{chunk['page']}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]


class ChromaVectorView:
    """A query view limited to the documents selected for the current request."""

    is_chroma_view = True

    def __init__(self, store: "ChromaVectorStore", doc_hashes: List[str]):
        self.store = store
        self.doc_hashes = doc_hashes

    def search(self, query_embedding: QueryEmbedding, top_k: int) -> List[EmbeddingChunk]:
        return self.store.query(
            query_embedding=query_embedding,
            doc_hashes=self.doc_hashes,
            top_k=top_k,
        )


class ChromaVectorStore:
    """
    Persistent Chroma collection for course document chunks.

    The collection stores vectors, texts, and metadata together. Documents are
    keyed by (source, doc_hash), which enables incremental updates without
    rebuilding vectors for unchanged files.
    """

    def __init__(
        self,
        path: str = CHROMA_PATH,
        collection_name: str = "course_chunks",
        client: Any | None = None,
    ):
        self.path = path
        self.collection_name = collection_name
        self._client = client
        self._collection: Any | None = None

    @property
    def collection(self) -> Any:
        if self._collection is None:
            if self._client is None:
                os.makedirs(self.path, exist_ok=True)
                import chromadb

                self._client = chromadb.PersistentClient(path=self.path)

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        return self._collection

    def delete_source(self, source: str) -> None:
        """Delete every chunk stored for a source, regardless of content hash."""
        self.collection.delete(where={"source": {"$eq": source}})

    def upsert_document(
        self,
        doc_hash: str,
        source: str,
        embedded_chunks: List[EmbeddingChunk],
    ) -> None:
        """
        Replace all chunks for one source with the supplied version.

        按 source 整体替换，而不是只删除 (source, doc_hash) 对：
        文件内容更新后 hash 会变化，只删同 hash 的 chunk 会让旧版本的
        向量残留在集合里，检索时返回已被替换掉的旧片段。
        """
        if not embedded_chunks:
            raise ValueError("embedded_chunks 不能为空")

        self.delete_source(source)

        ids = [_chunk_id(source, doc_hash, chunk) for chunk in embedded_chunks]
        documents = [str(chunk["text"]) for chunk in embedded_chunks]
        embeddings = [chunk["embedding"] for chunk in embedded_chunks]

        metadatas: List[Dict[str, Any]] = []
        for chunk in embedded_chunks:
            metadata: Dict[str, Any] = {
                "chunk_id": int(chunk["chunk_id"]),
                "page": int(chunk["page"]),
                "source": str(chunk["source"]),
                "doc_hash": doc_hash,
            }
            if chunk.get("heading"):
                metadata["heading"] = str(chunk["heading"])
            # 图片描述 chunk 携带原图路径，前端召回时可展示原图
            if chunk.get("image_path"):
                metadata["image_path"] = str(chunk["image_path"])
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_document(self, doc_hash: str, source: str) -> None:
        """Delete one logical document from the persistent collection."""
        self.collection.delete(
            where={
                "$and": [
                    {"doc_hash": {"$eq": doc_hash}},
                    {"source": {"$eq": source}},
                ]
            }
        )

    def has_document(self, doc_hash: str, source: str) -> bool:
        result = self.collection.get(
            where={
                "$and": [
                    {"doc_hash": {"$eq": doc_hash}},
                    {"source": {"$eq": source}},
                ]
            },
            limit=1,
            include=["metadatas"],
        )
        return bool(result.get("ids"))

    def get_documents(self, doc_hashes: List[str]) -> List[EmbeddingChunk]:
        """Return chunk metadata for the selected documents."""
        if not doc_hashes:
            return []

        result = self.collection.get(
            where={"doc_hash": {"$in": doc_hashes}},
            include=["documents", "metadatas"],
        )

        chunks: List[EmbeddingChunk] = []
        for document, metadata in zip(
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            chunk = dict(metadata)
            chunk["text"] = document
            chunks.append(chunk)

        return chunks

    def query(
        self,
        query_embedding: QueryEmbedding,
        doc_hashes: List[str],
        top_k: int,
    ) -> List[EmbeddingChunk]:
        """Query only the selected documents using cosine similarity."""
        if not doc_hashes or top_k <= 0:
            return []

        result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            where={"doc_hash": {"$in": doc_hashes}},
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[EmbeddingChunk] = []
        documents = result.get("documents", [[]])[0]
        metadata_groups = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for document, metadata, distance in zip(documents, metadata_groups, distances):
            chunk = dict(metadata)
            chunk["text"] = document
            # Chroma cosine distance = 1 - cosine similarity.
            chunk["score"] = 1.0 - float(distance)
            chunks.append(chunk)

        return chunks

    def for_documents(self, doc_hashes: List[str]) -> ChromaVectorView:
        return ChromaVectorView(self, doc_hashes)
