from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm

from src.config import (
    ALLOW_FILENAME_IMAGE_FALLBACK,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ENABLE_IMAGE_CAPTION,
    ENABLE_OCR,
    IMAGE_DIR,
    PDF_DIR,
    PROCESSED_DIR,
    ensure_dirs,
)
from src.llm_client import get_llm_client
from src.utils import chunk_text, clean_text, infer_topic_from_filename, write_json, write_jsonl
from src.vector_store import get_vector_store

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = f"[PDF extraction error on page {page_index}: {exc}]"
        text = clean_text(text)
        if text:
            pages.append((page_index, text))
    return pages


def build_text_chunks() -> List[Dict[str, Any]]:
    pdf_files = sorted([p for p in PDF_DIR.glob("**/*") if p.suffix.lower() == ".pdf"])
    chunks: List[Dict[str, Any]] = []
    counter = 0
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        topic = infer_topic_from_filename(pdf_path.name)
        for page_num, page_text in extract_pdf_pages(pdf_path):
            page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk_index, content in enumerate(page_chunks, start=1):
                counter += 1
                chunks.append(
                    {
                        "id": f"text_{counter:05d}",
                        "content": content,
                        "metadata": {
                            "source": pdf_path.name,
                            "source_path": str(pdf_path.relative_to(PDF_DIR.parent.parent)),
                            "page": page_num,
                            "chunk_index": chunk_index,
                            "modality": "text",
                            "topic": topic,
                        },
                    }
                )
    return chunks


def extract_ocr_text(image_path: Path) -> str:
    if not ENABLE_OCR:
        return ""
    try:
        import pytesseract

        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang="eng")
        return clean_text(text)
    except Exception:
        # Tesseract binary may not be installed on marker machines. VLM caption remains primary.
        return ""


def caption_image(image_path: Path) -> str:
    if not ENABLE_IMAGE_CAPTION:
        return ""
    try:
        llm = get_llm_client()
        caption, _usage = llm.caption_image(image_path)
        return clean_text(caption)
    except Exception as exc:
        if ALLOW_FILENAME_IMAGE_FALLBACK:
            return f"Image file related to {infer_topic_from_filename(image_path.name)}. Filename: {image_path.name}. Caption failed: {exc}"
        raise RuntimeError(
            f"Failed to caption image {image_path}. Configure VISION_MODEL_NAME/API key, "
            "or set ALLOW_FILENAME_IMAGE_FALLBACK=true for local debugging only."
        ) from exc


def build_image_chunks() -> List[Dict[str, Any]]:
    image_files = sorted([p for p in IMAGE_DIR.glob("**/*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS])
    chunks: List[Dict[str, Any]] = []
    counter = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        topic = infer_topic_from_filename(image_path.name)
        ocr_text = extract_ocr_text(image_path)
        caption = caption_image(image_path)
        content_parts = [
            f"Image filename: {image_path.name}.",
            f"Inferred topic: {topic}.",
        ]
        if caption:
            content_parts.append(f"Vision caption: {caption}")
        if ocr_text:
            content_parts.append(f"OCR text: {ocr_text}")
        content = clean_text("\n".join(content_parts))
        if not content:
            continue
        counter += 1
        chunks.append(
            {
                "id": f"image_{counter:05d}",
                "content": content,
                "metadata": {
                    "source": image_path.name,
                    "source_path": str(image_path.relative_to(IMAGE_DIR.parent.parent)),
                    "page": -1,
                    "chunk_index": 1,
                    "modality": "image",
                    "topic": topic,
                    "image_path": str(image_path),
                },
            }
        )
    return chunks


def build_index(reset: bool = True) -> Dict[str, Any]:
    ensure_dirs()
    text_chunks = build_text_chunks()
    image_chunks = build_image_chunks()
    all_chunks = text_chunks + image_chunks

    write_jsonl(PROCESSED_DIR / "text_chunks.jsonl", text_chunks)
    write_jsonl(PROCESSED_DIR / "image_chunks.jsonl", image_chunks)
    write_jsonl(PROCESSED_DIR / "all_chunks.jsonl", all_chunks)

    metadata = {
        "project": "A3 Java Programming Assistant",
        "domain": "personalised multimodal Java programming question answering",
        "num_pdf_files": len(list(PDF_DIR.glob("**/*.pdf"))),
        "num_image_files": len([p for p in IMAGE_DIR.glob("**/*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]),
        "num_text_chunks": len(text_chunks),
        "num_image_chunks": len(image_chunks),
        "num_all_chunks": len(all_chunks),
        "modalities": ["text", "image"],
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    write_json(PROCESSED_DIR / "metadata.json", metadata)

    store = get_vector_store()
    if reset:
        store.reset()
    store.add_chunks(all_chunks)
    metadata["chroma_count"] = store.count()
    write_json(PROCESSED_DIR / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma index from PDF and image data.")
    parser.add_argument("--no-reset", action="store_true", help="Append to existing Chroma collection instead of resetting it.")
    args = parser.parse_args()
    metadata = build_index(reset=not args.no_reset)
    print("\nIngestion completed.")
    for key, value in metadata.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
