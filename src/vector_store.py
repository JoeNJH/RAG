from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_BATCH_SIZE, ensure_dirs
from src.llm_client import get_llm_client
from src.utils import safe_metadata


class VectorStore:
    """Chroma wrapper for metadata-rich text and image-derived chunks."""

    def __init__(self, persist_directory: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME) -> None:
        ensure_dirs()
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Personalised multimodal Java programming KB"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Personalised multimodal Java programming KB"},
        )

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 10) -> None:
        if not chunks:
            return
        llm = get_llm_client()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            ids = [str(item["id"]) for item in batch]
            documents = [item.get("content", "") for item in batch]
            metadatas = [safe_metadata(item.get("metadata", {})) for item in batch]
            embeddings = llm.embed_texts(documents, batch_size=EMBEDDING_BATCH_SIZE)
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    def query(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        llm = get_llm_client()
        query_embedding = llm.embed_texts([query])[0]
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        rows: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for idx, doc_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else None
            score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
            metadata = metas[idx] if idx < len(metas) else {}
            rows.append(
                {
                    "id": doc_id,
                    "content": docs[idx] if idx < len(docs) else "",
                    "metadata": metadata,
                    "distance": distance,
                    "score": score,
                    "source": metadata.get("source", ""),
                    "modality": metadata.get("modality", ""),
                }
            )
        return rows


_default_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store
