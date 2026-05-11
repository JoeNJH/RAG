import os
from pathlib import Path
from dotenv import load_dotenv

# Project root: one level above src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# OpenAI-compatible mode for Aliyun Bailian / DashScope.
# Keep API key and base URL unchanged; switch models by changing model names.
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL_NAME = os.getenv("MODEL_NAME", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "qwen-vl-plus")

# For text-embedding-v4/text-embedding-v3, DashScope supports dimensions.
# Leave empty in .env to use the provider default.
_raw_dim = os.getenv("EMBEDDING_DIMENSIONS", "1024").strip()
EMBEDDING_DIMENSIONS = int(_raw_dim) if _raw_dim else None

CHROMA_DIR = str((PROJECT_ROOT / os.getenv("CHROMA_DIR", "./chroma_db")).resolve())
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "java_programming_kb")

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PDF_DIR = DATA_RAW_DIR / "pdfs"
IMAGE_DIR = DATA_RAW_DIR / "images"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = PROJECT_ROOT / "eval"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
# DashScope text-embedding-v4 currently accepts at most 10 input texts per embedding request.
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))

ENABLE_IMAGE_CAPTION = os.getenv("ENABLE_IMAGE_CAPTION", "true").lower() == "true"
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
USE_LLM_ROUTER = os.getenv("USE_LLM_ROUTER", "true").lower() == "true"
USE_LLM_VERIFIER = os.getenv("USE_LLM_VERIFIER", "true").lower() == "true"
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "true").lower() == "true"
ALLOW_FILENAME_IMAGE_FALLBACK = os.getenv("ALLOW_FILENAME_IMAGE_FALLBACK", "false").lower() == "true"

MAX_REPAIR_ATTEMPTS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "1"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "900"))


def validate_api_config(required: bool = True) -> bool:
    """Validate API config. Return True if usable; raise when required=True."""
    ok = bool(API_KEY and BASE_URL)
    if required and not ok:
        raise RuntimeError(
            "Missing API configuration. Please create .env based on .env.example "
            "and set OPENAI_API_KEY or DASHSCOPE_API_KEY."
        )
    return ok


def ensure_dirs() -> None:
    for directory in [PDF_DIR, IMAGE_DIR, PROCESSED_DIR, OUTPUTS_DIR, Path(CHROMA_DIR)]:
        directory.mkdir(parents=True, exist_ok=True)
