"""LM Studio embedding client via litellm (handles lm_studio/ prefix routing).

On first use, queries the API server for:
  - context_length: determines per-text character truncation limit
  - embedding dimension: used by vectordb for LanceDB schema creation

Both values are cached after the first call.
"""

import threading

from openai import OpenAI
from litellm import embedding
from config import API_BASE, EMBED_MODEL

BATCH_SIZE = 64
CHARS_PER_TOKEN = 1.5  # conservative for Korean-heavy text

_embed_config: dict | None = None  # {"max_chars": int, "dim": int}
_embed_lock = threading.Lock()


def get_context_length(model: str, api_base: str, fallback: int) -> int:
    """Query LM Studio for a model's context_length. Returns fallback on any failure."""
    try:
        client = OpenAI(base_url=api_base, api_key="lm-studio")
        model_id = model.removeprefix("lm_studio/")
        for m in client.models.list().data:
            if m.id == model_id or model_id in m.id or m.id in model_id:
                # Pydantic v2 stores unknown fields in model_extra; fall back to getattr
                extra = getattr(m, "model_extra", None) or {}
                ctx = extra.get("context_length") or getattr(m, "context_length", None)
                if ctx:
                    return int(ctx)
    except Exception:
        pass
    return fallback


def _init_embed() -> dict:
    """Detect embedding model's context length and output dimension from the API.

    Makes one minimal test embedding call to measure the actual vector dimension,
    so LanceDB schema creation never relies on a hardcoded value.
    Cached after the first call.
    """
    global _embed_config
    if _embed_config is not None:
        return _embed_config

    with _embed_lock:
        if _embed_config is not None:  # re-check after acquiring lock
            return _embed_config

        # 1. Context length → char limit
        token_limit = get_context_length(EMBED_MODEL, API_BASE, fallback=512)
        max_chars = max(int(token_limit * CHARS_PER_TOKEN) - 10, 50)

        # 2. Actual vector dimension via a minimal test call
        try:
            test = embedding(model=EMBED_MODEL, input=["test"], api_base=API_BASE)
            dim = len(test.data[0]["embedding"])
        except Exception:
            from config import EMBED_DIM  # env-override fallback
            dim = EMBED_DIM

        _embed_config = {"max_chars": max_chars, "dim": dim}
    return _embed_config


def embed_max_chars() -> int:
    """Max chars per text based on the embedding model's context window."""
    return _init_embed()["max_chars"]


def embed_dim() -> int:
    """Actual embedding vector dimension, detected from the API on first call."""
    return _init_embed()["dim"]


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts via LM Studio. Truncates each text to the model's context limit."""
    max_chars = embed_max_chars()
    truncated = [t[:max_chars] for t in texts]
    results: list[list[float]] = []

    for i in range(0, len(truncated), BATCH_SIZE):
        batch = truncated[i : i + BATCH_SIZE]
        response = embedding(model=EMBED_MODEL, input=batch, api_base=API_BASE)
        batch_results = sorted(response.data, key=lambda x: x["index"])
        results.extend(item["embedding"] for item in batch_results)

    return results
