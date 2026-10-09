"""Question paper generation: textbook chunks -> LLM (JSON) -> validated paper -> saved JSON."""
import datetime
import json
import random
import re
import uuid

import config
from src.rag.llm import chat_json
from src.rag.retriever import get_context_chunks

TYPES = {
    "mcq": {"label": "Multiple Choice Questions", "batch": 5,
            "desc": "multiple choice questions with exactly 4 options and one correct option"},
    "very_short": {"label": "Very Short Answer Questions", "batch": 5,
                   "desc": "very short answer questions (answer in 1-2 sentences)"},
    "short": {"label": "Short Answer Questions", "batch": 4,
              "desc": "short answer questions (answer in 3-5 sentences)"},
    "long": {"label": "Long Answer Questions", "batch": 3,
             "desc": "long answer / descriptive questions needing a detailed, step-by-step answer"},
    "case": {"label": "Case / Source Based Questions", "batch": 2,
             "desc": "case-based questions: each begins with a short original passage or data case "
                     "(3-5 lines) followed by one question on it"},
}

SYSTEM = """You are an experienced CBSE board examination paper setter.
Write original, exam-quality questions strictly based on the concepts in the TEXTBOOK CONTEXT.
Rules:
- Questions must be self-contained. Never mention 'the context', 'the passage', 'the text above' or page numbers.
- Match the requested difficulty and the marks each question carries.
- Always give a correct, concise model answer (for long answers include marking points).
- Do not repeat questions listed under ALREADY USED.
- Output ONLY valid JSON in the exact format requested."""

LANGS = {"same": "the same language as the textbook context", "English": "English",
         "Hindi": "Hindi (Devanagari script)", "Marathi": "Marathi (Devanagari script)"}


def default_sections():
    return [{"type": "mcq", "count": 10, "marks": 1}, {"type": "very_short", "count": 5, "marks": 2},
            {"type": "short", "count": 5, "marks": 3}, {"type": "long", "count": 3, "marks": 5}]


def _parse(text, qtype):
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except Exception:
            data = {}
    items = data.get("questions") if isinstance(data, dict) else data
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        q = str(it.get("question") or it.get("q") or "").strip()
        a = it.get("answer", "")
        a = "\n".join(map(str, a)) if isinstance(a, list) else str(a).strip()
        if not q or not a:
            continue
        item = {"question": q, "answer": a}
        if qtype == "mcq":
            opts = it.get("options")
            if isinstance(opts, dict):
                opts = list(opts.values())
            if not isinstance(opts, list) or len(opts) < 4:
                continue
            item["options"] = [re.sub(r"^\s*\(?[A-Da-d][\).:\-]\s+", "", str(o)).strip() for o in opts[:4]]
        out.append(item)
    return out


def _key(q):
    return re.sub(r"\W+", "", q.lower())[:60]


def generate_paper(params):
    """params: class, subject, sections[{type,count,marks}], difficulty, language,
    topics[list], school, title, duration. Yields ('progress', str) ... ('paper', dict)."""
    class_no, subject = int(params["class"]), params["subject"]
    diff = params.get("difficulty", "mixed")
    lang = LANGS.get(params.get("language", "same"), LANGS["same"])
    topics = [t.strip() for t in params.get("topics", []) if t.strip()]
    rng = random.Random()
    used, sections, qno = [], [], 0
    total_marks = sum(s["count"] * s["marks"] for s in params["sections"])

    for si, sec in enumerate(params["sections"]):
        t = TYPES[sec["type"]]
        name = f"Section {chr(65 + si)}"
        collected, attempts = [], 0
        while len(collected) < sec["count"] and attempts < 5:
            attempts += 1
            n = min(t["batch"], sec["count"] - len(collected))
            yield "progress", f"{name}: writing questions {len(collected) + 1}-{len(collected) + n} of {sec['count']}…"
            chunks = get_context_chunks(class_no, subject, topics, config.PAPER_CONTEXT_CHUNKS, rng)
            if not chunks:
                raise RuntimeError("No textbook content found for this class/subject. Run the ingestion first.")
            ctx = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))
            fmt = ('{"questions":[{"question":"...","options":["...","...","...","..."],"answer":"B) <correct option text>"}]}'
                   if sec["type"] == "mcq" else '{"questions":[{"question":"...","answer":"..."}]}')
            user = (f"TEXTBOOK CONTEXT:\n{ctx}\n\nALREADY USED:\n" + ("\n".join(f"- {u}" for u in used[-25:]) or "- none") +
                    f"\n\nTASK: Write {n} {t['desc']} for CBSE Class {class_no} {subject}.\n"
                    f"Difficulty: {diff}. Each question carries {sec['marks']} mark(s).\n"
                    f"Write in {lang}.\nReturn JSON only, in this format:\n{fmt}")
            try:
                got = _parse(chat_json(SYSTEM, user, config.PAPER_TEMPERATURE), sec["type"])
            except Exception as e:
                yield "progress", f"{name}: retrying after error ({e})"
                continue
            seen = {_key(u) for u in used}
            for g in got:
                if _key(g["question"]) in seen or len(collected) >= sec["count"]:
                    continue
                seen.add(_key(g["question"]))
                used.append(g["question"])
                collected.append(g)
        for g in collected:
            qno += 1
            g["no"] = qno
        sections.append({
            "name": name, "type": sec["type"], "label": t["label"], "marks_each": sec["marks"],
            "count": sec["count"], "questions": collected,
            "warning": None if len(collected) == sec["count"] else
            f"Only {len(collected)} of {sec['count']} questions could be generated.",
        })

    paper = {
        "id": uuid.uuid4().hex[:12],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "meta": {"school": params.get("school", ""), "title": params.get("title") or "Question Paper",
                 "class": class_no, "subject": subject, "difficulty": diff,
                 "duration": params.get("duration") or 90, "total_marks": total_marks,
                 "topics": topics},
        "sections": sections,
    }
    config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    (config.PAPERS_DIR / f"{paper['id']}.json").write_text(json.dumps(paper, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
    yield "paper", paper
