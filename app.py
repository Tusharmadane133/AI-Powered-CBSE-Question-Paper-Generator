import inspect
import json
import threading
from flask import Flask, Response, jsonify, render_template, request
import config
from src.rag.llm import check_ollama
from src.rag.rag_pipeline import answer_stream
app = Flask(__name__)
_gen_lock = threading.Lock()
_SUPPORTS_HISTORY = "history" in inspect.signature(answer_stream).parameters
def _clean_history(raw):
    out = []
    if not isinstance(raw, list):
        return out
    for m in raw[-6:]:
        if not isinstance(m, dict):
            continue
        role = "user" if m.get("role") == "user" else "assistant"
        content = str(m.get("content") or "").strip()[:600]
        if content:
            out.append({"role": role, "content": content})
    return out
def _with_context(q, history):
    """Add previous context only when the new question is clearly a follow-up."""
    if not history:
        return q
    ql = q.lower().strip()
    follow_up = (
        ql.startswith((
            "what is it", "what are they", "what does it", "why is it",
            "why does it", "how does it", "how is it", "how are they",
            "explain it", "explain this", "explain that", "describe it",
            "define it", "give an example", "give examples", "more about",
            "tell me more", "what about", "and what", "and why", "and how"
        ))
        or ql in {
            "it", "this", "that", "this one", "that one", "same",
            "same one", "why", "how", "more"
        }
        or ql.startswith(("this ", "that ", "these ", "those ", "same "))
    )
    if not follow_up:
        return q
    prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    return f"{prev[:300]}\n{q}" if prev else q
@app.get("/")
def index():
    return render_template("index.html", app_name=getattr(config, "APP_NAME", "CBSE Study Assistant"))
@app.get("/c/<chat_id>")
def chat_page(chat_id):
    return index()
@app.get("/api/health")
def health():
    try:
        ok = bool(check_ollama())
    except Exception:
        ok = False
    return jsonify(ollama=ok, model=getattr(config, "LLM_MODEL", getattr(config, "OLLAMA_MODEL", "")))
@app.get("/api/subjects")
def subjects():
    try:
        from src.rag.retriever import list_subjects
        with _gen_lock:
            data = list_subjects()
        return jsonify(data)
    except Exception as e:
        return jsonify(error=str(e)), 500
@app.post("/api/ask")
def ask():
    try:
        d = request.get_json(force=True) or {}
    except Exception:
        return jsonify(error="Invalid JSON request"), 400
    q = (d.get("question") or "").strip()
    if not q:
        return jsonify(error="Empty question"), 400
    class_no = d.get("class") or None
    subject = d.get("subject") or None
    book = d.get("book") or d.get("textbook") or d.get("selectedBook") or d.get("selected_textbook") or None
    if isinstance(book, str):
        book = book.strip() or None
    history = _clean_history(d.get("history"))
    def gen():
        _gen_lock.acquire()
        held = True
        try:
            if _SUPPORTS_HISTORY:
                stream = answer_stream(q, class_no, subject, book, history=history)
            else:
                stream = answer_stream(_with_context(q, history), class_no, subject, book)
            for kind, val in stream:
                if held:
                    _gen_lock.release()
                    held = False
                yield json.dumps({"type": kind, "data": val}, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}, ensure_ascii=False) + "\n"
        finally:
            if held:
                _gen_lock.release()
    return Response(gen(), mimetype="application/x-ndjson; charset=utf-8", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False, threaded=True)
