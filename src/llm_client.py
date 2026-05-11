import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

from src.config import (
    API_KEY,
    BASE_URL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    MAX_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
    VISION_MODEL_NAME,
    validate_api_config,
)


class LLMClient:
    """Thin wrapper around the OpenAI-compatible API.

    This project is configured for Aliyun Bailian/DashScope OpenAI-compatible mode.
    The same API key and base URL can be used for chat, embedding, and vision models;
    only the model names differ.
    """

    def __init__(self) -> None:
        validate_api_config(required=True)
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> Tuple[str, Dict[str, int]]:
        response = self.client.chat.completions.create(
            model=model or MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        return text, usage_dict

    def embed_texts(self, texts: Sequence[str], batch_size: Optional[int] = None) -> List[List[float]]:
        embeddings: List[List[float]] = []
        clean_texts = [t if t and t.strip() else "empty" for t in texts]

        # DashScope/OpenAI-compatible embedding endpoint has a strict batch-size limit.
        # text-embedding-v4 returns: "batch size ... should not be larger than 10"
        # when more than 10 input strings are sent in one request.
        effective_batch_size = min(batch_size or EMBEDDING_BATCH_SIZE, 10)

        for start in range(0, len(clean_texts), effective_batch_size):
            batch = clean_texts[start : start + effective_batch_size]
            kwargs: Dict[str, Any] = {
                "model": EMBEDDING_MODEL,
                "input": batch,
                "encoding_format": "float",
            }
            if EMBEDDING_DIMENSIONS:
                kwargs["dimensions"] = EMBEDDING_DIMENSIONS
            response = self.client.embeddings.create(**kwargs)
            # Ensure returned order follows input order.
            data = sorted(response.data, key=lambda item: item.index)
            embeddings.extend([item.embedding for item in data])
        return embeddings

    def caption_image(self, image_path: str | Path) -> Tuple[str, Dict[str, int]]:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_b64}"

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are helping build a personalised multimodal Java programming knowledge base. "
                    "Describe the image as searchable evidence. Focus on Java, Spring Boot, IDE, error messages, "
                    "architecture layers, diagrams, arrows, labels, and any visible text. Be concise but specific."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Generate a detailed caption for this image. Include visible error messages, module names, "
                            "class names, architecture relationships, and likely topic keywords."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        return self.chat(messages=messages, model=VISION_MODEL_NAME, temperature=0.1, max_tokens=500)


_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
