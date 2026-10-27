import inspect
import json
import threading
import time
import uuid
from pathlib import Path
from src.rag.embeddings import warmup
warmup()
from flask import Flask, Response, jsonify, render_template, request, send_file

import config
from src.rag.llm import check_ollama
from src.rag.rag_pipeline import answer_stream

app = Flask(__name__)

# Serializes access to the RAG pipeline if the retriever/model is not thread-safe.
_gen_lock = threading.Lock()
_SUPPORTS_HISTORY = "history" in inspect.signature(answer_stream).parameters

JOBS = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL = 3600


def _line(kind, val):
    return json.dumps({"type": kind, "data": val}, ensure_ascii=False) + "\n"


def _cleanup_jobs():
    now = time.time()
    with _JOBS_LOCK:
        expired = [
            key for key, value in JOBS.items()
            if value["done"] and now - value["t"] > _JOB_TTL
        ]
        for key in expired:
            JOBS.pop(key, None)


def _run_job(job_id, gen):
    job = JOBS[job_id]
    try:
        for line in gen:
            job["lines"].append(line)
    except Exception as exc:
        job["lines"].append(_line("error", str(exc)))
    finally:
        job["t"] = time.time()
        job["done"] = True


def _stream_job(job_id, start=0):
    job = JOBS.get(job_id)
    if not job:
        yield _line("error", "Answer expired, please ask again.")
        return

    index = start
    while True:
        while index < len(job["lines"]):
            yield job["lines"][index]
            index += 1
        if job["done"] and index >= len(job["lines"]):
            break
        time.sleep(0.05)


def _ndjson_response(gen):
    return Response(
        gen,
        mimetype="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _clean_history(raw):
    out = []
    if not isinstance(raw, list):
        return out

    for message in raw[-6:]:
        if not isinstance(message, dict):
            continue
        role = "user" if message.get("role") == "user" else "assistant"
        content = str(message.get("content") or "").strip()[:600]
        if content:
            out.append({"role": role, "content": content})
    return out


def _with_context(question, history):
    if not history:
        return question

    q_lower = question.lower().strip()
    follow_up = (
        q_lower.startswith((
            "what is it", "what are they", "what does it", "why is it",
            "why does it", "how does it", "how is it", "how are they",
            "explain it", "explain this", "explain that", "describe it",
            "define it", "give an example", "give examples", "more about",
            "tell me more", "what about", "and what", "and why", "and how",
        ))
        or q_lower in {
            "it", "this", "that", "this one", "that one", "same",
            "same one", "why", "how", "more",
        }
        or q_lower.startswith(("this ", "that ", "these ", "those ", "same "))
    )

    if not follow_up:
        return question

    previous_question = next(
        (message["content"] for message in reversed(history)
         if message["role"] == "user"),
        "",
    )
    return f"{previous_question[:300]}\n{question}" if previous_question else question


def _pdf_roots():
    roots = []
    for value in (
        getattr(config, "BOOKS_DIR", None),
        getattr(config, "DATA_DIR", None),
        Path(__file__).resolve().parent / "data" / "books",
        Path(__file__).resolve().parent / "data",
    ):
        if value:
            path = Path(value)
            if path.is_dir() and path not in roots:
                roots.append(path)
    return roots


def _normalise_name(value):
    return "".join(char.lower() for char in str(value or "") if char.isalnum())


def _pdf_text_for_matching(path, max_pages=None):
    """Extract PDF text to match chapter citation names to whole-book PDFs."""
    try:
        import fitz  # PyMuPDF
        with fitz.open(path) as document:
            page_count = len(document) if max_pages is None else min(len(document), max_pages)
            return " ".join(
                document[index].get_text("text") for index in range(page_count)
            ).lower()
    except Exception:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        page_count = len(reader.pages) if max_pages is None else min(len(reader.pages), max_pages)
        return " ".join(
            reader.pages[index].extract_text() or "" for index in range(page_count)
        ).lower()
    except Exception:
        return ""


def _find_source_pdf(book):
    """Find a PDF by filename first, then by chapter/book title in the PDF text."""
    wanted = _normalise_name(book)
    if not wanted:
        return None

    files = []
    for root in _pdf_roots():
        try:
            files.extend(path for path in root.rglob("*.pdf") if path.is_file())
        except (OSError, PermissionError):
            continue

    if not files:
        return None

    for path in files:
        if _normalise_name(path.stem) == wanted:
            return path

    filename_matches = [
        path for path in files
        if wanted in _normalise_name(path.stem)
        or _normalise_name(path.stem) in wanted
    ]
    if len(filename_matches) == 1:
        return filename_matches[0]

    text_matches = []
    for path in files:
        content = _normalise_name(_pdf_text_for_matching(path))
        if wanted and wanted in content:
            text_matches.append(path)

    if len(text_matches) == 1:
        return text_matches[0]
    if len(filename_matches) > 1:
        return None
    return text_matches[0] if len(text_matches) == 1 else None


@app.get("/")
def index():
    return render_template(
        "index.html",
        app_name=getattr(config, "APP_NAME", "CBSE Study Assistant"),
    )


@app.get("/chat/<chat_id>")
@app.get("/c/<chat_id>")
def chat_page(chat_id):
    return index()


@app.get("/api/health")
def health():
    try:
        ollama_ok = bool(check_ollama())
    except Exception:
        ollama_ok = False

    return jsonify(
        ollama=ollama_ok,
        model=getattr(config, "LLM_MODEL", getattr(config, "OLLAMA_MODEL", "")),
    )


@app.get("/api/subjects")
def subjects():
    try:
        from src.rag.retriever import list_subjects
        with _gen_lock:
            data = list_subjects()
        return jsonify(data)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.get("/api/source")
def api_source():
    book = (request.args.get("book") or "").strip()
    try:
        page = max(1, int(request.args.get("page", "1")))
    except (TypeError, ValueError):
        page = 1

    pdf = _find_source_pdf(book)
    if not pdf:
        return jsonify(
            error="PDF not found for this source. Check the book title and PDF filename in data/books."
        ), 404

    response = send_file(
        pdf,
        mimetype="application/pdf",
        as_attachment=False,
        conditional=True,
        download_name=pdf.name,
    )
    response.headers["Content-Disposition"] = f'inline; filename="{pdf.name}"'
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Source-Page"] = str(page)
    return response


@app.get("/api/stream/<job_id>")
def api_stream(job_id):
    try:
        start = max(0, int(request.args.get("from", 0)))
    except (TypeError, ValueError):
        start = 0
    return _ndjson_response(_stream_job(job_id, start))


@app.post("/api/ask")
def ask():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify(error="Invalid JSON request"), 400

    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="Empty question"), 400

    class_no = data.get("class") or None
    subject = data.get("subject") or None
    book = (
        data.get("book")
        or data.get("textbook")
        or data.get("selectedBook")
        or data.get("selected_textbook")
        or None
    )
    if isinstance(book, str):
        book = book.strip() or None

    history = _clean_history(data.get("history"))

    def gen():
        # Server-side timing: includes time waiting for the generation lock and
        # the complete answer_stream execution. The browser timer measures the
        # end-to-end time seen by the student.
        request_started = time.perf_counter()
        lock_started = time.perf_counter()
        lock_wait_seconds = 0.0
        first_event_seconds = None
        event_count = 0
        lock_held = False

        try:
            _gen_lock.acquire()
            lock_held = True
            lock_wait_seconds = time.perf_counter() - lock_started

            if _SUPPORTS_HISTORY:
                stream = answer_stream(
                    question, class_no, subject, book, history=history
                )
            else:
                contextual_question = _with_context(question, history)
                stream = answer_stream(
                    contextual_question, class_no, subject, book
                )

            for kind, value in stream:
                if first_event_seconds is None:
                    first_event_seconds = time.perf_counter() - request_started

                event_count += 1
                yield _line(kind, value)

        except Exception as exc:
            yield _line("error", str(exc))
        finally:
            if lock_held:
                _gen_lock.release()

            total_seconds = time.perf_counter() - request_started
            first_event = (
                round(first_event_seconds, 3)
                if first_event_seconds is not None else None
            )
            metrics = {
                "total_seconds": round(total_seconds, 3),
                "lock_wait_seconds": round(lock_wait_seconds, 3),
                "time_to_first_event_seconds": first_event,
                "stream_events": event_count,
            }

            # Visible in the Flask terminal for performance diagnosis.
            print(
                "[RAG_TIMING] "
                f"total={metrics['total_seconds']:.3f}s "
                f"lock_wait={metrics['lock_wait_seconds']:.3f}s "
                f"first_event={metrics['time_to_first_event_seconds']}s "
                f"events={metrics['stream_events']}",
                flush=True,
            )

            # The frontend can optionally display this event. Older frontends
            # ignore unknown event types, so the existing stream remains usable.
            yield _line("timing", metrics)

    _cleanup_jobs()
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "lines": [_line("job", job_id)],
        "done": False,
        "t": time.time(),
    }

    threading.Thread(
        target=_run_job,
        args=(job_id, gen()),
        daemon=True,
    ).start()

    return _ndjson_response(_stream_job(job_id, 0))


if __name__ == "__main__":
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )