import json

import requests

import config


def check_ollama() -> bool:
    try:
        return requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


def _repeat_penalty():
    # RAG answers must copy words from the textbook text. A repeat penalty above ~1.0 punishes
    # exactly those words and produces garbage like "toggleodd" and "factors.a locked an".
    # Endless loops are already stopped by _is_looping() in rag_pipeline.py, so keep this at 1.0.
    return min(float(getattr(config, "REPEAT_PENALTY", 1.0)), 1.02)


def _temperature(value=None):
    t = value if value is not None else config.LLM_TEMPERATURE
    return min(float(t), 0.2)  # textbook answers must be factual, not creative


def stream_chat(system, user):
    """Yield tokens from Ollama /api/chat."""
    payload = {
        "model": config.LLM_MODEL,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": _temperature(),
            "top_p": 0.9,
            "num_ctx": config.LLM_NUM_CTX,
            "repeat_penalty": _repeat_penalty(),   # light: still stops "word word word" loops
            "repeat_last_n": 64,                   # only look at recent words, not the whole text
            "num_predict": getattr(config, "MAX_TOKENS", 1000),  # hard cap on answer length
        },
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    with requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, stream=True,
                       timeout=config.LLM_TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            tok = data.get("message", {}).get("content", "")
            if tok:
                yield tok
            if data.get("done"):
                break


def chat_json(system, user, temperature=None):
    """Non-streaming call that forces valid JSON output."""
    payload = {
        "model": config.LLM_MODEL, "stream": False, "format": "json",
        "options": {"temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
                    "num_ctx": config.LLM_NUM_CTX},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=config.LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"]