"""Cached sentence-transformer embeddings with lightweight timing diagnostics."""
from functools import lru_cache
import logging
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    """Load the embedding model once per Python process and reuse it."""
    started = time.perf_counter()
    model_name = config.EMBED_MODEL

    thread_value = getattr(config, "EMBED_CPU_THREADS", os.getenv("EMBED_CPU_THREADS"))
    if thread_value:
        try:
            thread_count = max(1, int(thread_value))
            try:
                import torch
                torch.set_num_threads(thread_count)
            except (ImportError, RuntimeError):
                logger.debug("Could not set PyTorch thread count", exc_info=True)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid EMBED_CPU_THREADS value: %r", thread_value)

    # Local cache first: no HF Hub network check. Download only if not cached.
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        logger.info("Embedding model not cached locally; downloading %s", model_name)
        model = SentenceTransformer(model_name)
    logger.info(
        "[RAG_TIMING] embedding_model_load=%.3fs model=%s",
        time.perf_counter() - started,
        model_name,
    )
    return model


def _encode(texts, *, batch_size):
    """Encode texts with the existing E5 query/passage prefix convention."""
    if not texts:
        return np.empty((0, config.EMBED_DIM), dtype=np.float32)

    model = _model()
    started = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    logger.info(
        "[RAG_TIMING] embedding_encode=%.3fs count=%d batch_size=%d",
        time.perf_counter() - started,
        len(texts),
        batch_size,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_passages(texts):
    """Embed document passages using the E5 passage prefix."""
    if texts is None:
        texts = []
    if not texts:
        return np.empty((0, config.EMBED_DIM), dtype=np.float32)

    batch_size = max(1, int(getattr(config, "EMBED_BATCH_SIZE", 32)))
    return _encode([f"passage: {text}" for text in texts], batch_size=batch_size)


def embed_query(text):
    """Embed a query using the E5 query prefix; model remains cached."""
    started = time.perf_counter()
    vector = _encode([f"query: {text}"], batch_size=1)[0]
    logger.info(
        "[RAG_TIMING] embed_query_total=%.3fs model_cache=%s",
        time.perf_counter() - started,
        _model.cache_info(),
    )
    return vector


def warmup():
    """Call once at app start: loads embedding model, index and Ollama model."""
    import requests
    from src.rag import retriever
    from src.rag.llm import _chat_options

    try:
        embed_query("warm up")
        retriever._client()
        retriever._corpus()
        retriever._lexical_index()
        retriever.list_subjects()
        retriever.retrieve("what is a fraction", top_k=2)
        opts = _chat_options()
        opts["num_predict"] = 1
        requests.post(
            f"{config.OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": config.LLM_MODEL,
                "stream": False,
                "keep_alive": "30m",
                "options": opts,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=180,
        )
        print("[WARMUP] ready", flush=True)
    except Exception as exc:
        print(f"[WARMUP] failed: {exc!r}", flush=True)