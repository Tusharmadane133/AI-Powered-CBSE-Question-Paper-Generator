import json

import requests

import config


def check_ollama() -> bool:
    try:
        return requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


def stream_chat(system, user):
    """Yield tokens from Ollama /api/chat."""
    payload = {
        "model": config.LLM_MODEL,
        "stream": True,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_ctx": config.LLM_NUM_CTX,
            "repeat_penalty": getattr(config, "REPEAT_PENALTY", 1.15),  # stops "word word word" loops
            "num_predict": getattr(config, "MAX_TOKENS", 800),          # hard cap on answer length
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