import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata values must be scalar: str/int/float/bool."""
    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = json.dumps(value, ensure_ascii=False)
    return cleaned


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path, default: Optional[Any] = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-based chunker suitable for English/Chinese mixed notes.

    It tries to cut on paragraph boundaries while keeping implementation simple.
    """
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]
        # Prefer paragraph/sentence boundary near the end.
        cut_candidates = [window.rfind("\n\n"), window.rfind(". "), window.rfind("。"), window.rfind("; ")]
        cut = max(cut_candidates)
        if cut > chunk_size * 0.55 and end < len(text):
            end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def infer_topic_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower().replace("_", " ").replace("-", " ")
    topic_keywords = {
        "spring": "Spring Boot",
        "boot": "Spring Boot",
        "maven": "Maven dependency",
        "mysql": "MySQL database",
        "jdbc": "JDBC",
        "mybatis": "MyBatis",
        "collection": "Java collections",
        "arraylist": "Java collections",
        "linkedlist": "Java collections",
        "hashmap": "Java collections",
        "jvm": "JVM",
        "thread": "Java concurrency",
        "concurrency": "Java concurrency",
        "error": "Java debugging",
        "exception": "Java debugging",
        "controller": "Spring Boot architecture",
        "service": "Spring Boot architecture",
        "mapper": "Spring Boot architecture",
        "architecture": "Software architecture",
        "uml": "UML diagram",
    }
    for key, topic in topic_keywords.items():
        if key in stem:
            return topic
    return stem.title() if stem else "Java programming"


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from an LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None
