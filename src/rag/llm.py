"""Ollama client helpers with first-token and total-generation timing logs."""
import json
import logging
import time

import requests

import config

logger = logging.getLogger(__name__)


def check_ollama() -> bool:
    """Return True when the local Ollama server responds to /api/tags."""
    try:
        return requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


def _repeat_penalty() -> float:
    # Avoid penalizing normal textbook phrasing too strongly.
    return min(float(getattr(config, "REPEAT_PENALTY", 1.0)), 1.02)


def _temperature(value=None) -> float:
    """Keep generation conservative for textbook-grounded answers."""
    temperature = value if value is not None else getattr(config, "LLM_TEMPERATURE", 0.1)
    return min(float(temperature), 0.2)


def _num_predict(default=220) -> int:
    """Use a moderate default answer length; config.MAX_TOKENS overrides it."""
    try:
        value = int(getattr(config, "MAX_TOKENS", default))
    except (TypeError, ValueError):
        value = default
    return max(64, min(value, 1200))


def _chat_options(temperature=None):
    """Share consistent generation settings between streaming and JSON calls."""
    options = {
        "temperature": _temperature(temperature),
        "top_p": 0.9,
        "num_ctx": int(getattr(config, "LLM_NUM_CTX", 4096)),
        "repeat_penalty": _repeat_penalty(),
        "repeat_last_n": 64,
        "num_predict": _num_predict(),
    }
    return options


def stream_chat(system, user):
    """Yield Ollama response chunks and log time-to-first-token and total duration."""
    payload = {
        "model": config.LLM_MODEL,
        "stream": True,
        "keep_alive": "30m",
        "options": _chat_options(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    started = time.perf_counter()
    first_token_at = None
    token_chunks = 0
    output_chars = 0
    try:
        with requests.post(
            f"{config.OLLAMA_URL.rstrip('/')}/api/chat",
            json=payload,
            stream=True,
            timeout=getattr(config, "LLM_TIMEOUT", (10, 180)),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.warning("[RAG_TIMING] Ignoring malformed Ollama stream line")
                    continue

                token = data.get("message", {}).get("content", "")
                if token:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        logger.info(
                            "[RAG_TIMING] llm_first_token=%.3fs model=%s",
                            first_token_at - started,
                            config.LLM_MODEL,
                        )
                    token_chunks += 1
                    output_chars += len(token)
                    yield token

                if data.get("done"):
                    break
    except Exception:
        logger.exception("[RAG_TIMING] llm_stream_error")
        raise
    finally:
        elapsed = time.perf_counter() - started
        first_elapsed = (first_token_at - started) if first_token_at is not None else None
        logger.info(
            "[RAG_TIMING] llm_generation=%.3fs first_token=%s chunks=%d chars=%d model=%s",
            elapsed,
            f"{first_elapsed:.3f}s" if first_elapsed is not None else "none",
            token_chunks,
            output_chars,
            config.LLM_MODEL,
        )


def chat_json(system, user, temperature=None):
    """Make a non-streaming Ollama call that requests valid JSON."""
    payload = {
        "model": config.LLM_MODEL,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": _chat_options(temperature),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    started = time.perf_counter()
    try:
        response = requests.post(
            f"{config.OLLAMA_URL.rstrip('/')}/api/chat",
            json=payload,
            timeout=getattr(config, "LLM_TIMEOUT", (10, 180)),
        )
        response.raise_for_status()
        body = response.json()
        result = body.get("message", {}).get("content", "")
        if not isinstance(result, str):
            raise ValueError("Ollama returned an invalid JSON response body")
        logger.info(
            "[RAG_TIMING] llm_json=%.3fs chars=%d model=%s",
            time.perf_counter() - started,
            len(result),
            config.LLM_MODEL,
        )
        return result
    except Exception:
        logger.exception("[RAG_TIMING] llm_json_error")
        raise
