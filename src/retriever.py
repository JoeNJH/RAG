from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from src.config import TOP_K
from src.vector_store import get_vector_store

RetrievalMode = Literal["text", "image", "hybrid"]


def _dedupe_by_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for item in sorted(items, key=lambda x: x.get("score", 0), reverse=True):
        item_id = item.get("id")
        if item_id in seen:
            continue
        seen.add(item_id)
        output.append(item)
    return output


class Retriever:
    """Retrieval strategy layer.

    Modes:
    - text: only PDF-derived text chunks.
    - image: only OCR/caption image chunks.
    - hybrid: text + image-derived evidence.
    """

    def __init__(self) -> None:
        self.store = get_vector_store()

    def retrieve(self, query: str, mode: RetrievalMode = "hybrid", top_k: int = TOP_K) -> List[Dict[str, Any]]:
        if mode == "text":
            return self.store.query(query, top_k=top_k, where={"modality": "text"})
        if mode == "image":
            return self.store.query(query, top_k=top_k, where={"modality": "image"})

        # Hybrid retrieval keeps both modalities visible for analysis and ablation.
        text_k = max(2, top_k // 2 + 1)
        image_k = max(2, top_k // 2 + 1)
        text_results = self.store.query(query, top_k=text_k, where={"modality": "text"})
        image_results = self.store.query(query, top_k=image_k, where={"modality": "image"})
        merged = _dedupe_by_id(text_results + image_results)
        if len(merged) < top_k:
            # Backfill from all chunks if one modality is sparse.
            all_results = self.store.query(query, top_k=top_k)
            merged = _dedupe_by_id(merged + all_results)
        return merged[:top_k]


_default_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever


def format_evidence(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "No retrieved evidence."
    lines = []
    for i, item in enumerate(evidence, start=1):
        meta = item.get("metadata", {}) or {}
        source = meta.get("source", item.get("source", "unknown"))
        page = meta.get("page", "")
        modality = meta.get("modality", item.get("modality", "unknown"))
        topic = meta.get("topic", "")
        score = item.get("score", 0.0)
        location = f", page {page}" if page not in (None, "", -1, "-1") else ""
        lines.append(
            f"[{i}] Source: {source}{location} | Modality: {modality} | Topic: {topic} | Score: {score:.3f}\n"
            f"{item.get('content', '').strip()}"
        )
    return "\n\n".join(lines)
