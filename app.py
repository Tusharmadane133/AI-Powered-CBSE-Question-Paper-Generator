import inspect
import json
import threading
import time
import uuid
from flask import Flask, Response, jsonify, render_template, request
import config
from src.rag.llm import check_ollama
from src.rag.rag_pipeline import answer_stream

app = Flask(__name__)
_gen_lock = threading.Lock()
_SUPPORTS_HISTORY = "history" in inspect.signature(answer_stream).parameters

# ---------- background jobs: answer keeps generating even if the browser reloads ----------
JOBS = {}  # job_id -> {"lines": [...], "done": bool, "t": float}
_JOBS_LOCK = threading.Lock()
_JOB_TTL = 3600  # seconds


def _line(kind, val):
    return json.dumps({"type": kind, "data": val}, ensure_ascii=False) + "\n"


def _cleanup_jobs():
    now = time.time()
    with _JOBS_LOCK:
        for k in [k for k, v in JOBS.items() if v["done"] and now - v["t"] > _JOB_TTL]:
            JOBS.pop(k, None)


def _run_job(job_id, gen):
    job = JOBS[job_id]
    try:
        for line in gen:
            job["lines"].append(line)
    except Exception as e:
        job["lines"].append(_line("error", str(e)))
    job["t"] = time.time()
    job["done"] = True


def _stream_job(job_id, start=0):
    job = JOBS.get(job_id)
    if not job:
        yield _line("error", "Answer expired, please ask again.")
        return
    i = start
    while True:
        while i < len(job["lines"]):
            yield job["lines"][i]
            i += 1
        if job["done"]:
            if i >= len(job["lines"]):
                break
            continue
        time.sleep(0.05)


def _ndjson_response(gen):
    return Response(
        gen,
        mimetype="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


@app.get("/chat/<chat_id>")
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


@app.get("/api/stream/<job_id>")
def api_stream(job_id):
    try:
        start = max(0, int(request.args.get("from", 0)))
    except ValueError:
        start = 0
    return _ndjson_response(_stream_job(job_id, start))


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
                yield _line(kind, val)
        except Exception as e:
            yield _line("error", str(e))
        finally:
            if held:
                _gen_lock.release()

    _cleanup_jobs()
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"lines": [_line("job", job_id)], "done": False, "t": time.time()}
    threading.Thread(target=_run_job, args=(job_id, gen()), daemon=True).start()
    return _ndjson_response(_stream_job(job_id, 0))


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False, threaded=True)