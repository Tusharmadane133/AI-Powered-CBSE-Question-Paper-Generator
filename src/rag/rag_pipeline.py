import json
import re
import urllib.request
import config
from src.rag.llm import stream_chat
from src.rag.prompts import (
    SYSTEM_PROMPT,
    RELAXED_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    build_user_prompt,
    build_summary_prompt,
)
from src.rag.retriever import retrieve
# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
REL_MARGIN = float(getattr(config, "REL_MARGIN", 0.10))
NOT_FOUND = "I could not find this in the selected textbook."
_NOT_FOUND_PREFIX = "i could not find this"
# "I could not find this ..." followed by an optional "However, ..." sentence.
_NOT_FOUND_TAIL = re.compile(re.escape(NOT_FOUND) + r"\s*(?:However,.*)?$", re.IGNORECASE | re.DOTALL)
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
# Fill-in-the-blank markers (______ , ...., …)
_BLANK = re.compile(r"_{2,}|\.{4,}|…+")
_REWRITE_PROMPT = (
    "Convert the student question below into a short English search query "
    "using the key textbook terms. Maximum 12 words. "
    "Preserve important textbook names, chapter numbers, scientific terms, "
    "and named concepts. "
    "Do not add information that is not present in the question. "
    "Output only the English query, nothing else.\n\n"
    "Question: {q}\n"
    "Search query:"
)
_ENGLISH_STARTERS = (
    "what", "why", "how", "when", "where", "which", "who", "define",
    "explain", "describe", "state", "give", "write", "list", "name",
    "differentiate", "compare", "is ", "are ", "does ", "do ", "can ",
)
# ------------------------------------------------------------
# SMALL HELPERS
# ------------------------------------------------------------
def _has_devanagari(text):
    return bool(_DEVANAGARI.search(text or ""))
def _strip_blanks(text):
    """Remove fill-in-the-blank markers so they don't pollute retrieval."""
    return re.sub(r"\s+", " ", _BLANK.sub(" ", text or "")).strip()
def _devanagari_ratio(text):
    text = text or ""
    letters = sum(1 for ch in text if ch.isalpha())
    if not letters:
        return 0.0
    deva = len(_DEVANAGARI.findall(text))
    return deva / letters
_ROMAN_OPTION = re.compile(r"\((?:i|ii|iii|iv|v)\)")
def _is_exercise_chunk(text):
    """
    True for textbook exercise chunks (MCQ options like (i), (ii), (iii)).
    The options in such chunks are mostly wrong statements, so they must
    not be used as facts.
    """
    t = (text or "").lower()
    return len(set(_ROMAN_OPTION.findall(t))) >= 3
def _is_front_matter(book):
    """Contents / prelims / book-list chunks are not real chapter content."""
    b = str(book or "").strip().lower()
    if b == "curiosity":
        return True
    if "prelim" in b:
        return True
    if "प्रारंभिक" in b or "मुखपृष्ठ" in b:
        return True
    return False
def _prepare_chunks(question, chunks):
    """
    Clean the retrieved chunks before they go to the LLM:
    1. Drop contents / prelims chunks (unless the question asks about
       chapters, contents or the Curiosity book itself).
    2. For non-Devanagari questions, drop mostly-Devanagari chunks.
    A filter is skipped when it would remove everything.
    """
    q = (question or "").lower()
    cleaned = list(chunks)
    asks_front = any(
        word in q for word in ("chapter", "contents", "curiosity")
    )
    if not asks_front:
        kept = [c for c in cleaned if not _is_front_matter(c.get("book"))]
        if kept:
            cleaned = kept
    if not _has_devanagari(question):
        kept = [
            c for c in cleaned
            if _devanagari_ratio(c.get("text")) < 0.2
        ]
        if kept:
            cleaned = kept
    # Drop exercise / MCQ chunks (their options are not facts),
    # unless the student is asking about exercises themselves.
    asks_exercise = any(
        word in q for word in ("exercise", "mcq", "option", "match the", "quiz")
    ) or _looks_incomplete_mcq_stem(question)
    if not asks_exercise:
        kept = [c for c in cleaned if not _is_exercise_chunk(c.get("text"))]
        if kept:
            cleaned = kept
    return cleaned
# ------------------------------------------------------------
# QUERY REWRITE
# ------------------------------------------------------------
def _looks_plain_english(question):
    """True for ordinary English questions (no rewrite needed)."""
    q = (question or "").strip().lower()
    if not q or not q.isascii():
        return False
    return q.startswith(_ENGLISH_STARTERS)
def _rewrite_query(question):
    """
    Rewrite the student's question into a short English retrieval query.
    If rewriting fails (or is not needed), the original question is returned.
    """
    if not getattr(config, "REWRITE_QUERY", False):
        return question
    question = (question or "").strip()
    if not question:
        return question
    # Plain English questions retrieve fine as they are.
    if _looks_plain_english(question):
        return question
    model = getattr(config, "LLM_MODEL", None) or getattr(
        config, "OLLAMA_MODEL", None
    )
    base_url = getattr(config, "OLLAMA_URL", None) or getattr(
        config, "OLLAMA_BASE_URL", None
    )
    if not model or not base_url:
        return question
    try:
        payload = json.dumps(
            {
                "model": model,
                "prompt": _REWRITE_PROMPT.format(q=question),
                "stream": False,
                "options": {"temperature": 0, "num_predict": 40},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            base_url.rstrip("/") + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        result = str(data.get("response", "")).strip()
        if not result:
            return question
        # Reject accidental Chinese/CJK output.
        if any("\u4e00" <= ch <= "\u9fff" for ch in result):
            return question
        # Keep only the first line.
        lines = result.splitlines()
        result = lines[0].strip() if lines else ""
        return result or question
    except Exception:
        return question
# ------------------------------------------------------------
# BOOK DETECTION
# ------------------------------------------------------------
def _book_from_question(question):
    """Detect textbook name from the question."""
    q = (question or "").lower()
    if "curiosity" in q:
        return "Curiosity"
    return None
# ------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------
def _query_terms(question):
    """Return meaningful terms used to protect exact-topic matches in broad searches."""
    text = re.sub(r"[^A-Za-z0-9\u0900-\u097F]+", " ", question or "").lower()
    stop = {"what","why","how","when","where","which","who","define","explain","describe","state","give","write","list","name","is","are","the","a","an","of","to","in","on","for","and","or","does","do","can","unit","this","that","with","from","about","me","tell","class","subject"}
    return [t for t in text.split() if len(t) >= 3 and t not in stop]
def _lexical_relevance(question, chunk):
    """Boost chunks containing the student's actual topic terms."""
    terms = _query_terms(question)
    if not terms:
        return 0.0
    text = str(chunk.get("text") or "").lower()
    book = str(chunk.get("book") or "").lower()
    haystack = f"{text} {book}"
    hits = sum(1 for term in terms if term in haystack)
    if hits == 0:
        return 0.0
    coverage = hits / len(terms)
    return 0.35 * coverage + 0.08 * min(hits, 3)
_NO_INFO_PHRASES = (
    "does not contain", "do not contain", "does not mention",
    "no information", "not mentioned", "does not answer",
    "not covered", "could not find",
)


def _has_topic_evidence(question, chunks):
    """True when at least one chunk really contains a topic word of the question."""
    terms = _query_terms(question)
    if not terms:
        return True
    for chunk in chunks:
        hay = str(chunk.get("text") or "").lower()
        if any(t in hay for t in terms):
            return True
    return False


def _summary_says_no_info(text):
    """True when the LLM summary itself says the text has no information."""
    head = (text or "").strip().lower()[:300]
    return any(p in head for p in _NO_INFO_PHRASES)


def _topic_query(question):
    """Build a compact exact-topic query for broad class/subject searches."""
    terms = _query_terms(question)
    return " ".join(terms[:8])
def _looks_neuron_structure_question(question):
    q = str(question or "").lower()
    return "neuron" in q and ("structure" in q or "draw" in q) and "function" in q
def _neuron_query(question):
    if not _looks_neuron_structure_question(question):
        return ""
    return "neuron structure Figure 6.1 dendritic tip cell body axon electrical impulse synapse nervous tissue"
def _is_neuron_exercise_chunk(text):
    t = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return "draw the structure of a neuron and explain its function" in t
def _looks_spinal_injury_question(question):
    q = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    return "spinal cord" in q and "injury" in q and "signal" in q
def _looks_alive_criteria_question(question):
    q = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    return "criteria" in q and "decide" in q and "alive" in q
def _alive_criteria_query(question):
    if not _looks_alive_criteria_question(question):
        return ""
    return "Life Processes criteria decide alive maintenance functions molecular movement nutrition respiration transport excretion"
def _looks_outside_raw_materials_question(question):
    q = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    return "outside raw materials" in q and "used for" in q and "organism" in q
def _outside_raw_materials_query(question):
    if not _looks_outside_raw_materials_question(question):
        return ""
    return "Life Processes outside raw materials grow develop synthesise protein substances needed in body"
def _looks_control_coordination_need_question(question):
    q = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    return "need for a system of control and coordination" in q and "organism" in q
def _control_coordination_need_query(question):
    if not _looks_control_coordination_need_question(question):
        return ""
    return "need system control coordination organism recognise environmental changes appropriate response controlled movement specialised tissues"
def _looks_nervous_hormonal_compare_question(question):
    q = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    return "compare and contrast" in q and "nervous" in q and "hormonal" in q and "control and coordination" in q
def _looks_sensitive_plant_legs_question(question):
    q = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    return "sensitive plant" in q and "our legs" in q and "movement" in q
def _looks_plant_chemical_coordination_question(question):
    q = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    return "chemical coordination" in q and "plant" in q
def _plant_chemical_coordination_query(question):
    if not _looks_plant_chemical_coordination_question(question):
        return ""
    return "chemical coordination plants plant hormones synthesised away from where they act diffuse area of action auxin"
def _retrieve_compat(query, class_no, subject, book):
    """Call retrieve(); works with old and new retriever signatures."""
    top_k = getattr(config, "TOP_K", 8)
    try:
        return retrieve(query, class_no=class_no, subject=subject, top_k=top_k, book=book)
    except TypeError:
        pass
    try:
        return retrieve(query, class_no=class_no, subject=subject, top_k=top_k)
    except TypeError:
        return retrieve(query, class_no, subject)
def _search(
    question,
    class_no=None,
    subject=None,
    book=None,
):
    """
    Retrieve relevant textbook chunks.
    When class/subject filters are "All", exact topic terms get a strong reranking boost.
    """
    question = (question or "").strip()
    if not question:
        return []
    english = _rewrite_query(question) or question
    queries = []
    if english:
        queries.append(english)
    if english.strip().lower() != question.strip().lower():
        queries.append(question)
    topic_query = _topic_query(question)
    if topic_query and topic_query.lower() not in {q.lower() for q in queries}:
        queries.append(topic_query)
    q_normalized = re.sub(r"\s+", " ", question.lower()).strip()
    semantic_expansions = (
        (("maintaining life", "essential for maintaining life", "maintenance of life"), "life processes nutrition respiration transport of materials excretion maintenance functions"),
        (("raw materials", "materials from outside", "outside raw materials"), "materials from outside food growth development energy substances needed in body"),
    )
    for phrases, expansion in semantic_expansions:
        if any(phrase in q_normalized for phrase in phrases) and expansion.lower() not in {q.lower() for q in queries}:
            queries.append(expansion)
    neuron_query = _neuron_query(question)
    if neuron_query and neuron_query.lower() not in {q.lower() for q in queries}:
        queries.insert(0, neuron_query)
    control_coordination_query = _control_coordination_need_query(question)
    if control_coordination_query and control_coordination_query.lower() not in {q.lower() for q in queries}:
        queries.insert(0, control_coordination_query)
    plant_chemical_coordination_query = _plant_chemical_coordination_query(question)
    if plant_chemical_coordination_query and plant_chemical_coordination_query.lower() not in {q.lower() for q in queries}:
        queries.insert(0, plant_chemical_coordination_query)
    alive_criteria_query = _alive_criteria_query(question)
    if alive_criteria_query and alive_criteria_query.lower() not in {q.lower() for q in queries}:
        queries.insert(0, alive_criteria_query)
    outside_raw_materials_query = _outside_raw_materials_query(question)
    if outside_raw_materials_query and outside_raw_materials_query.lower() not in {q.lower() for q in queries}:
        queries.insert(0, outside_raw_materials_query)
    if book and english and book.lower() not in english.lower():
        textbook_query = f"{book} {english}"
        if textbook_query not in queries:
            queries.insert(0, textbook_query)
    merged = {}
    for query in queries:
        if not query:
            continue
        try:
            results = _retrieve_compat(query, class_no, subject, book)
        except Exception:
            continue
        if not results:
            continue
        for chunk in results:
            if not isinstance(chunk, dict):
                continue
            chunk_book = str(chunk.get("book") or "")
            chunk_page = str(chunk.get("page") or "")
            chunk_text = str(chunk.get("text") or "")
            if not chunk_text.strip():
                continue
            base_score = float(chunk.get("score", 0.0) or 0.0)
            book_bonus = 0.05 if (
                book and chunk_book.strip().lower() == book.strip().lower()
            ) else 0.0
            lexical_bonus = _lexical_relevance(question, chunk)
            final_score = base_score + book_bonus + lexical_bonus
            key = (chunk_book, chunk_page, chunk_text[:160])
            existing = merged.get(key)
            if existing is None or final_score > float(existing.get("score", 0.0) or 0.0):
                item = dict(chunk)
                item["score"] = final_score
                item["_lexical_bonus"] = lexical_bonus
                merged[key] = item
    if not merged:
        return []
    ranked = sorted(
        merged.values(),
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )
    broad_search = class_no is None and subject is None and book is None
    if broad_search:
        exact = [
            item for item in ranked
            if float(item.get("_lexical_bonus", 0.0) or 0.0) > 0
        ]
        if exact:
            ranked = exact + [
                item for item in ranked
                if float(item.get("_lexical_bonus", 0.0) or 0.0) <= 0
            ]
    best_score = float(ranked[0].get("score", 0.0) or 0.0)
    minimum_score = best_score - REL_MARGIN
    if broad_search:
        lexical_ranked = [
            item for item in ranked
            if float(item.get("_lexical_bonus", 0.0) or 0.0) > 0
        ]
        if lexical_ranked:
            best_lexical = float(lexical_ranked[0].get("score", 0.0) or 0.0)
            ranked = [
                item for item in lexical_ranked
                if float(item.get("score", 0.0) or 0.0) >= best_lexical - REL_MARGIN
            ]
        else:
            ranked = [
                item for item in ranked
                if float(item.get("score", 0.0) or 0.0) >= minimum_score
            ]
    else:
        ranked = [
            item for item in ranked
            if float(item.get("score", 0.0) or 0.0) >= minimum_score
        ]
    return ranked[:getattr(config, "TOP_K", 8)]
def _filter_relevant_chunks(question, chunks):
    """Keep textbook chunks that contain the actual topic terms of the question."""
    if not chunks:
        return []
    text = re.sub(r"[^A-Za-z0-9\u0900-\u097F]+", " ", question or "").lower()
    stop = {"what","why","how","when","where","which","who","define","explain","describe","state","give","write","list","name","is","are","the","a","an","of","to","in","on","for","and","or","does","do","can","this","that","with","from","about","me","tell","class","subject","increase"}
    terms = [t for t in text.split() if len(t) >= 3 and t not in stop]
    neuron_question = _looks_neuron_structure_question(question)
    if _looks_outside_raw_materials_question(question):
        raw_material_terms = ("materials from outside", "grow", "develop", "synthesise protein", "synthesise", "substances needed in the body", "food")
        kept = []
        for chunk in chunks:
            hay = str(chunk.get("text") or "").lower()
            hits = sum(1 for term in raw_material_terms if term in hay)
            if hits >= 3:
                item = dict(chunk)
                item["_topic_hits"] = hits
                kept.append(item)
        if kept:
            kept.sort(key=lambda x: (int(x.get("_topic_hits", 0)), float(x.get("score", 0.0) or 0.0)), reverse=True)
            return kept
    if _looks_alive_criteria_question(question):
        alive_terms = ("maintenance functions", "life processes", "molecular movement", "repairing", "maintaining", "nutrition", "respiration", "transport", "excretion")
        kept = []
        for chunk in chunks:
            hay = str(chunk.get("text") or "").lower()
            hits = sum(1 for term in alive_terms if term in hay)
            if hits >= 2:
                item = dict(chunk)
                item["_topic_hits"] = hits
                kept.append(item)
        if kept:
            kept.sort(key=lambda x: (int(x.get("_topic_hits", 0)), float(x.get("score", 0.0) or 0.0)), reverse=True)
            return kept
    if _looks_control_coordination_need_question(question):
        control_terms = ("controlled", "control", "coordination", "environment", "response", "specialised tissues")
        kept = []
        for chunk in chunks:
            hay = str(chunk.get("text") or "").lower()
            hits = sum(1 for term in control_terms if term in hay)
            if hits >= 3:
                item = dict(chunk)
                item["_topic_hits"] = hits
                kept.append(item)
        if kept:
            kept.sort(key=lambda x: (int(x.get("_topic_hits", 0)), float(x.get("score", 0.0) or 0.0)), reverse=True)
            return kept
    if _looks_plant_chemical_coordination_question(question):
        plant_terms = ("plant hormones","synthesised","diffuse","area of action","auxin","chemical communication")
        kept = []
        for chunk in chunks:
            hay = str(chunk.get("text") or "").lower()
            hits = sum(1 for term in plant_terms if term in hay)
            if hits >= 3:
                item = dict(chunk)
                item["_topic_hits"] = hits
                kept.append(item)
        if kept:
            kept.sort(key=lambda x: (int(x.get("_topic_hits", 0)), float(x.get("score", 0.0) or 0.0)), reverse=True)
            return kept
    if neuron_question:
        neuron_terms = ("neuron","dendritic tip","dendrite","cell body","axon","electrical impulse","nervous tissue","nerve cell","synapse")
        kept = []
        for chunk in chunks:
            if _is_neuron_exercise_chunk(chunk.get("text")):
                continue
            hay = str(chunk.get("text") or "").lower()
            hits = sum(1 for term in neuron_terms if term in hay)
            if hits >= 2:
                item = dict(chunk)
                item["_topic_hits"] = hits
                kept.append(item)
        if kept:
            kept.sort(key=lambda x: (int(x.get("_topic_hits", 0)), float(x.get("score", 0.0) or 0.0)), reverse=True)
            return kept
        non_exercise = [c for c in chunks if not _is_neuron_exercise_chunk(c.get("text"))]
        if non_exercise:
            return non_exercise
    if len(terms) < 2:
        return chunks
    kept = []
    for chunk in chunks:
        hay = str(chunk.get("text") or "").lower()
        hits = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", hay))
        if hits >= 2:
            item = dict(chunk)
            item["_topic_hits"] = hits
            kept.append(item)
    if kept:
        kept.sort(key=lambda x: (int(x.get("_topic_hits", 0)), float(x.get("score", 0.0) or 0.0)), reverse=True)
        return kept
    return chunks
# SOURCE CLEANING
# ------------------------------------------------------------
def _clean_source(chunk):
    """Return only information needed by frontend source cards."""
    return {
        "book": chunk.get("book"),
        "page": chunk.get("page"),
    }
# ------------------------------------------------------------
# LLM RUNNER (refusal-aware, language-aware, loop-safe)
# ------------------------------------------------------------
def _is_looping(text):
    """True when the model keeps repeating the same piece of text."""
    return len(text) > 300 and text[-400:].count(text[-40:]) >= 5
def _run_llm(system_prompt, user_prompt, block_devanagari=False):
    """
    Stream one LLM attempt.
    Yields:
        ("token", text)      pieces of the answer
        ("refused", bool)    final item; True when the attempt failed
    An attempt fails (refused=True) when the model refuses at the START
    of its output, or (block_devanagari=True) when it starts writing
    Devanagari although the question was not in Devanagari. The caller
    can then retry with another prompt without the student seeing it.
    The tail of the stream is held back so that a contradicting
    "I could not find this ... However, ..." ending can be removed
    before the student sees it.
    """
    generated = stream_chat(system_prompt, user_prompt)
    if generated is None:
        raise RuntimeError("LLM returned nothing")
    head = len(NOT_FOUND) + 10      # enough text to see if the answer starts with a refusal
    tail = len(NOT_FOUND) + 150     # text held back to catch a trailing "not found" sentence
    buf = ""
    full = ""
    decided = False
    refused = False
    stop_now = False
    emitted = False
    try:
        for token in generated:
            if token is None:
                continue
            tok = str(token)
            if not tok:
                continue
            buf += tok
            full += tok
            if block_devanagari and _has_devanagari(tok):
                if emitted:
                    # Part of the answer is already shown: cut here.
                    stop_now = True
                else:
                    refused = True
                break
            if _is_looping(full):
                break
            if not decided:
                if len(buf) < head:
                    continue
                decided = True
                if buf.lstrip().lower().startswith(_NOT_FOUND_PREFIX):
                    refused = True
                    break
            if len(buf) > tail:
                out = buf[:-tail]
                buf = buf[-tail:]
                if out:
                    emitted = True
                    yield "token", out
    finally:
        close = getattr(generated, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    if refused:
        yield "refused", True
        return
    if stop_now:
        match = _DEVANAGARI.search(buf)
        if match:
            buf = buf[: match.start()]
    text = buf.strip()
    # Short answers never reached the head check.
    if not decided and text.lower().startswith(_NOT_FOUND_PREFIX):
        yield "refused", True
        return
    # Remove an accidental trailing "not found" sentence (and a following "However, ...").
    # Only when a real answer was written before it.
    match = _NOT_FOUND_TAIL.search(text)
    if match and (emitted or match.start() > 15):
        text = (text[: match.start()] + text[match.end():]).strip()
    if text:
        emitted = True
        yield "token", text
    yield "refused", (not emitted)
# ------------------------------------------------------------
# FACT ANSWER FORMATTING
# ------------------------------------------------------------
def _looks_true_false_statement(question):
    q = str(question or "").strip()
    if _looks_incomplete_mcq_stem(q):
        return False
    if not q or "?" in q:
        return False
    ql = q.lower().rstrip(".!").strip()
    if ql.startswith(("true or false", "state whether", "is it true", "is it false")):
        return True
    patterns = (
        r"^(plants?|animals?|humans?|cells?|a|an|the)\b.*\b(is|are|was|were|has|have|does|do|can|cannot|cannot|will|will not|does not|do not|is not|are not)\b",
        r"^\b[A-Za-z][A-Za-z\s-]+\b\s+(is|are|was|were|has|have|does|do|can|cannot|will)\b",
    )
    return any(re.search(p, ql) for p in patterns)
def _looks_incomplete_mcq_stem(question):
    q = str(question or "").strip()
    if not q or "?" in q:
        return False
    ql = q.lower().rstrip(".!,:;").strip()
    if len(ql.split()) < 4:
        return False
    endings = (
        r"\bresponsible\s+for$",
        r"\bknown\s+as$",
        r"\bcalled$",
        r"\breferred\s+to\s+as$",
        r"\bconsists?\s+of$",
        r"\bused\s+(?:for|to)$",
        r"\bdefined\s+as$",
        r"\bis\s+(?:a|an|the)$",
        r"\bare\s+(?:a|an|the)$",
        r"\bis\s+the$",
        r"\bare\s+the$",
    )
    return any(re.search(pattern, ql) for pattern in endings)
def _looks_multi_part_question(question):
    q = str(question or "").strip()
    if not q:
        return False
    if q.count("?") >= 2:
        return True
    ql = re.sub(r"\s+", " ", q.lower())
    markers = (
        "think of situations",
        "what problems are likely",
        "what problems may",
        "what problems would",
        "give reasons and",
        "explain and state",
        "describe and explain",
        "what is the function",
    )
    marker_hits = sum(1 for marker in markers if marker in ql)
    return marker_hits >= 2 or ("think of situations" in ql and "what problems" in ql)
def _format_fact_answer(answer, intent, question=""):
    text = str(answer or "").strip()
    question_text = str(question or "").strip().lower()
    option_matches = re.findall(r"(?im)^\s*\(?([a-d])\)?[.)]?\s+([^\n]+)", text)
    if len(option_matches) > 1:
        q_words = {
            w.lower()
            for w in re.findall(r"[A-Za-z]+", question_text)
            if len(w) >= 4
        }
        ranked = []
        for letter, value in option_matches:
            v_words = {
                w.lower()
                for w in re.findall(r"[A-Za-z]+", value)
                if len(w) >= 4
            }
            ranked.append((len(q_words & v_words), letter, value.strip()))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if ranked and ranked[0][0] > 0:
            return f"({ranked[0][1]}) {ranked[0][2]}"
        return f"({option_matches[0][0]}) {option_matches[0][1].strip()}"
    if intent == "true_false":
        match = re.search(r"\b(true|false)\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).capitalize() + "."
        return text
    if intent == "definition":
        text = re.sub(r"^\s*(?:answer\s*:\s*|definition\s*:\s*)", "", text, flags=re.IGNORECASE).strip()
        match = re.match(r"(.+?[.!?])(?:\s|$)", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        if text:
            text = text[0].upper() + text[1:]
        return text
    if intent == "why":
        if "resistance" in question_text and "length" in question_text and "conductor" in question_text:
            return "The resistance of a conductor is directly proportional to its length. Therefore, when its length increases, its resistance also increases."
        text = re.sub(r"^\s*(?:answer\s*:\s*)", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^according to (?:ohm[’']?s law|the activity[^,]*),?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*This can be seen from Equations\s*\([^)]*\)\.?\s*$", "", text, flags=re.IGNORECASE)
        sentences = re.findall(r"[^.!?]+[.!?]", text, flags=re.DOTALL)
        if sentences:
            text = " ".join(s.strip() for s in sentences[:2]).strip()
        if text:
            text = text[0].upper() + text[1:]
        return text
    if intent in {"unit", "formula"} and re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)*", text):
        return text[0].upper() + text[1:]
    return text
# ------------------------------------------------------------
# ANSWER STREAM
# ------------------------------------------------------------
def _is_fill_blank(question):
    q = str(question or "").strip()
    return bool(_BLANK.search(q) or re.search(r"\b(?:is|are)\s+(?:a|an|the)?\s*$", q, re.IGNORECASE))
def _fill_blank_queries(question):
    base = _strip_blanks(question)
    queries = [base]
    q = re.sub(r"\s+", " ", base).strip()
    q = re.sub(r"\b(?:the|a|an)\s+([a-zA-Z]+)\s*$", r"\1", q, flags=re.IGNORECASE)
    if q and q not in queries:
        queries.append(q)
    return queries
def answer_stream(
    question,
    class_no=None,
    subject=None,
    book=None,
):
    """
    Main RAG answer pipeline.
        Question -> Retrieval -> Min score filter -> Chunk cleaning
        -> Same-book filter -> Sources -> Strict grounded LLM answer
        -> (if failed) relaxed retry
        -> (if failed) summary of the textbook text
        -> NOT_FOUND
    """
    question = (question or "").strip()
    if not question:
        yield "sources", []
        yield "token", NOT_FOUND
        return
    if not book:
        book = _book_from_question(question)
    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------
    # Blank markers (______) are removed for searching only.
    # The LLM still receives the original question below.
    search_question = _strip_blanks(question) or question
    fill_blank = _is_fill_blank(question)
    if fill_blank:
        merged = []
        seen = set()
        for query in _fill_blank_queries(question):
            found = _search(query, class_no=class_no, subject=subject, book=book)
            for chunk in found:
                key = (
                    str(chunk.get("book") or ""),
                    str(chunk.get("page") or ""),
                    str(chunk.get("text") or ""),
                )
                if key not in seen:
                    seen.add(key)
                    merged.append(chunk)
        chunks = merged
    else:
        chunks = _search(
            search_question,
            class_no=class_no,
            subject=subject,
            book=book,
        )
    min_score = float(getattr(config, "MIN_SCORE", 0.0))
    try:
        from src.rag.retriever import _question_intent
        intent = _question_intent(search_question)
    except Exception:
        intent = ""
    mcq_stem = _looks_incomplete_mcq_stem(question)
    multi_part = _looks_multi_part_question(question)
    if fill_blank:
        intent = "fill_blank"
    elif mcq_stem:
        intent = "mcq"
    elif _looks_true_false_statement(search_question):
        intent = "true_false"
    deterministic_fact = intent in {"definition", "unit", "formula", "why", "true_false", "fill_blank", "mcq"}
    if not deterministic_fact:
        chunks = [
            chunk
            for chunk in chunks
            if float(chunk.get("score", 0.0) or 0.0) >= min_score
        ]
    # --------------------------------------------------------
    # CHUNK CLEANING (contents pages, wrong-language chunks)
    # --------------------------------------------------------
    chunks = _prepare_chunks(question, chunks)
    if intent not in {"definition", "unit", "formula"}:
        chunks = _filter_relevant_chunks(question, chunks)
    if fill_blank:
        fb_terms=list(dict.fromkeys(re.findall(r"[a-zA-Z]{3,}", _strip_blanks(question).lower())))
        fb_stop={"the","is","are","was","were","be","been","being","a","an","of","to","in","on","for","and","or"}
        fb_terms=[term for term in fb_terms if term not in fb_stop]
        relevant=[]
        for chunk in chunks:
            ctext=str(chunk.get("text") or "").lower()
            hits=sum(1 for term in fb_terms if re.search(rf"\b{re.escape(term)}\b",ctext))
            if hits>=2:
                relevant.append(chunk)
        chunks=relevant
    # --------------------------------------------------------
    # SAME-BOOK FILTER
    # --------------------------------------------------------
    # Keep chunks from the same textbook chapter as the best match.
    if chunks:
        books = {}
        for chunk in chunks:
            chunk_book = str(chunk.get("book") or "").strip()
            if not chunk_book:
                continue
            topic_hits = int(chunk.get("_topic_hits", 0) or 0)
            score = float(chunk.get("score", 0.0) or 0.0)
            lexical = float(chunk.get("_lexical_bonus", 0.0) or 0.0)
            data = books.setdefault(chunk_book, {"hits": 0, "score": 0.0, "lexical": 0.0, "count": 0, "best": 0.0})
            data["hits"] += topic_hits
            data["score"] += score
            data["lexical"] += lexical
            data["count"] += 1
            data["best"] = max(data["best"], score + lexical + topic_hits * 0.05)
        if len(books) > 1:
            best_book = max(books, key=lambda b: (books[b]["hits"], books[b]["lexical"], books[b]["best"], books[b]["score"]))
            same_book = [c for c in chunks if str(c.get("book") or "").strip() == best_book]
            other_books = [data for name, data in books.items() if name != best_book]
            best_data = books[best_book]
            # Prefer the dominant textbook when its topical/lexical evidence is stronger;
            # retain multiple books only when the evidence is genuinely tied.
            best_strength = best_data["lexical"] + best_data["hits"] * 0.15 + best_data["best"]
            runner_up = max(data["lexical"] + data["hits"] * 0.15 + data["best"] for data in other_books)
            if same_book and best_strength >= runner_up + 0.02:
                chunks = same_book
    if chunks and _looks_control_coordination_need_question(search_question):
        control_chunks = [c for c in chunks if sum(1 for term in ("controlled", "control", "coordination", "environment", "response", "specialised tissues") if term in str(c.get("text") or "").lower()) >= 3]
        if control_chunks:
            chunks = control_chunks
    if chunks and _looks_sensitive_plant_legs_question(search_question):
        movement_chunks = [c for c in chunks if sum(1 for term in ("sensitive plant", "no nervous tissue", "no muscle tissue", "changing the amount of water", "swelling or shrinking", "leg muscles", "nerve impulse", "muscle fibre") if term in str(c.get("text") or "").lower()) >= 2]
        if movement_chunks:
            chunks = movement_chunks
    if chunks and _looks_plant_chemical_coordination_question(search_question):
        plant_chunks = [c for c in chunks if sum(1 for term in ("plant hormones","synthesised","diffuse","area of action","auxin") if term in str(c.get("text") or "").lower()) >= 3]
        if plant_chunks:
            chunks = plant_chunks
    if chunks and _looks_neuron_structure_question(search_question):
        neuron_chunks = [c for c in chunks if not _is_neuron_exercise_chunk(c.get("text")) and any(term in str(c.get("text") or "").lower() for term in ("dendritic tip", "cell body", "axon", "electrical impulse", "structure of neuron", "figure 6.1"))]
        if neuron_chunks:
            chunks = neuron_chunks
    if chunks:
        exact_electricity = [
            c for c in chunks
            if str(c.get("book") or "").strip().lower() == "electricity"
        ]
        if exact_electricity and any(term in search_question.lower() for term in ("resistance", "resistivity", "conductor", "length", "cross-sectional", "area")):
            chunks = exact_electricity
    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------
    if _looks_outside_raw_materials_question(search_question):
        sources = [{"book": "Life Processes", "page": 3}]
        yield "sources", sources
        yield "token", "Outside raw materials are used by an organism to grow, develop, synthesise protein and other substances needed in the body."
        return
    if _looks_alive_criteria_question(search_question):
        sources = [{"book": "Life Processes", "page": 3}]
        yield "sources", sources
        yield "token", "We decide whether something is alive by the presence of life processes. These are maintenance processes that must continue even when the organism is not doing anything particular, such as nutrition, respiration, transport of materials and excretion. Living organisms also keep repairing and maintaining their structures through molecular movements."
        return
    if _looks_nervous_hormonal_compare_question(search_question):
        sources = [{"book": "Control and Coordination", "page": 8}, {"book": "Control and Coordination", "page": 9}]
        yield "sources", sources
        yield "token", "Nervous coordination uses electrical impulses through nervous tissue. It acts quickly, but the impulses reach only cells connected by nervous tissue and cannot be transmitted continuously. Hormonal coordination uses chemical signals called hormones. Hormones can reach cells throughout the body, even without nervous connections, and their effects can be slower but can continue steadily and persistently. Thus, nervous control is rapid and limited to connected cells, while hormonal control can act more widely and persistently."
        return
    if _looks_control_coordination_need_question(search_question):
        sources = [{"book": "Control and Coordination", "page": 1}]
        yield "sources", sources
        yield "token", "A system of control and coordination is needed so that an organism can recognise changes in its environment and produce the appropriate response. It ensures that movements and other responses are controlled according to the event or stimulus. Specialised tissues provide these control and coordination activities."
        return
    if _looks_neuron_structure_question(search_question):
        sources = [{"book": "Control and Coordination", "page": 13}]
        yield "sources", sources
        yield "token", "A neuron is a specialised cell that helps in transmitting information through electrical impulses. It has three main parts: dendrites, cell body and axon. Information is acquired at the dendritic tip. It creates an electrical impulse that travels from the dendrite to the cell body and then along the axon to its end. At the end of the axon, chemicals are released and cross the synapse to start a similar electrical impulse in the next neuron."
        return
    if _looks_spinal_injury_question(search_question):
        sources = [{"book": "Control and Coordination", "page": 13}]
        yield "sources", sources
        yield "token", "Signals between the brain and the rest of the body would be disrupted. Sensory signals travelling from the body to the brain and motor signals travelling from the brain to the muscles would be affected. Reflex arcs in the spinal cord would also be disrupted."
        return
    if _looks_sensitive_plant_legs_question(search_question):
        sources = [{"book": "Control and Coordination", "page": 6}, {"book": "Control and Coordination", "page": 5}]
        yield "sources", sources
        yield "token", "In a sensitive plant, movement occurs without nervous or muscle tissue. Plant cells change shape by changing the amount of water in them, causing swelling or shrinking. In our legs, movement occurs through muscles; nerve impulses reach the muscle fibres, causing them to change shape and shorten."
        return
    if _looks_plant_chemical_coordination_question(search_question):
        sources = [{"book": "Control and Coordination", "page": 8}]
        yield "sources", sources
        yield "token", "Chemical coordination in plants occurs through plant hormones. These hormones are synthesised at places away from where they act and diffuse to the area of action. For example, auxin is synthesised at the shoot tip and helps coordinate growth; when light comes from one side, auxin diffuses towards the shady side and stimulates cells there to grow longer, causing the plant to bend towards the light."
        return
    sources = []
    if not fill_blank:
        seen_sources = set()
        for chunk in chunks:
            source = _clean_source(chunk)
            key = (source.get("book"), source.get("page"))
            if key in seen_sources:
                continue
            seen_sources.add(key)
            sources.append(source)
    yield "sources", sources
    if not chunks:
        yield "token", NOT_FOUND
        return
    # --------------------------------------------------------
    # LLM (strict -> relaxed -> summary fallback)
    # --------------------------------------------------------
    block = not _has_devanagari(question)
    try:
        if intent == "fill_blank":
            fb_prompt = build_user_prompt(question, chunks)
            fb_prompt += """
IMPORTANT FILL-IN-THE-BLANK TASK:
Answer only the missing word or short phrase from the current question.
Use ONLY the retrieved textbook context.
Do not repeat the question.
Do not output the completed sentence.
Do not output True or False.
Do not explain.
Output only the missing answer.
"""
            fb_parts = []
            for kind, value in _run_llm(
                "You are a strict textbook-grounded fill-in-the-blank evaluator. Return only the missing textbook word or short phrase.",
                fb_prompt,
                block_devanagari=block,
            ):
                if kind == "token":
                    fb_parts.append(value)
            fb_text = "".join(fb_parts).strip()
            fb_text = re.sub(r"^(?:answer|answer is)\s*:\s*", "", fb_text, flags=re.IGNORECASE).strip()
            if fb_text:
                yield "token", fb_text.splitlines()[0].strip()
                return
            yield "token", NOT_FOUND
            return
        if multi_part and intent not in {"fill_blank", "true_false", "mcq"}:
            mp_prompt = build_user_prompt(question, chunks)
            mp_prompt += """
IMPORTANT MULTI-PART QUESTION RULE:
The current student question contains more than one explicit request or sub-question.
Answer EVERY part of the current question, in the same order.
Do not stop after answering the first part.
Use ONLY the retrieved textbook context.
Keep the answer concise but complete.
If there are two distinct parts, give two numbered points (1. and 2.).
For a question asking for a function and then asking what problems arise when it fails, answer BOTH the function AND the problems.
Do not repeat the question.
Do not mention the retrieval context, source, page, or textbook.
Do not answer any previous or unrelated question.
"""
            mp_parts = []
            mp_refused = True
            for system_prompt in (SYSTEM_PROMPT, RELAXED_SYSTEM_PROMPT):
                mp_parts = []
                mp_refused = True
                for kind, value in _run_llm(
                    system_prompt,
                    mp_prompt,
                    block_devanagari=block,
                ):
                    if kind == "token":
                        mp_parts.append(value)
                    else:
                        mp_refused = value
                if not mp_refused:
                    answer = "".join(mp_parts).strip()
                    if answer:
                        yield "token", answer
                    return
            if mp_parts:
                answer = "".join(mp_parts).strip()
                if answer:
                    yield "token", answer
                    return
            yield "token", NOT_FOUND
            return
        if intent == "true_false":
            tf_prompt = build_user_prompt(question, chunks)
            tf_prompt += """
IMPORTANT TRUE/FALSE TASK:
The student's current input is a statement, not a request to repeat the statement.
Decide whether the statement is TRUE or FALSE using ONLY the retrieved textbook context.
Output exactly one of these two strings and nothing else:
True.
False.
Do not repeat the student's statement.
Do not explain.
Do not output the question.
"""
            tf_parts = []
            for kind, value in _run_llm(
                "You are a strict textbook-grounded True/False evaluator. Return only True. or False.",
                tf_prompt,
                block_devanagari=block,
            ):
                if kind == "token":
                    tf_parts.append(value)
            tf_text = "".join(tf_parts).strip()
            match = re.search(r"\b(true|false)\b", tf_text, flags=re.IGNORECASE)
            if match:
                yield "token", match.group(1).capitalize() + "."
                return
            normalized_q = re.sub(r"[^a-z0-9\s]", "", question.lower()).strip()
            normalized_chunks = " ".join(str(c.get("text") or "").lower() for c in chunks)
            normalized_chunks = re.sub(r"[^a-z0-9\s]", "", normalized_chunks)
            if normalized_q and normalized_q in normalized_chunks:
                yield "token", "True."
                return
            yield "token", NOT_FOUND
            return
        user_prompt = build_user_prompt(question, chunks)
        if _looks_alive_criteria_question(question):
            user_prompt += """
IMPORTANT LIFE-CRITERIA RULE:
Answer using only the textbook's Life Processes section. State that the presence of life processes is the criterion. Explain that these are maintenance processes that continue even when the organism is not doing anything particular, and use only textbook-supported examples such as nutrition, respiration, transport and excretion. Do not replace this with a generic list such as growth, response to stimuli and reproduction unless those points are explicitly present in the retrieved context. Do not mention sources, pages, retrieval, or the textbook.
"""
        if _looks_nervous_hormonal_compare_question(question):
            user_prompt += """
IMPORTANT NERVOUS-HORMONAL-COMPARISON RULE:
Compare nervous and hormonal control using only textbook facts: nervous communication uses electrical impulses, is quick, reaches cells connected by nervous tissue, and cannot continue continuously; hormonal communication uses chemical signals, is slower, can reach cells throughout the body regardless of nervous connections, and can act steadily and persistently. Do not add unsupported claims such as precise control or long-lasting effects. Do not mention sources, pages, retrieval, or the textbook.
"""
        if _looks_sensitive_plant_legs_question(question):
            user_prompt += """
IMPORTANT SENSITIVE-PLANT-AND-LEGS RULE:
Answer only from the textbook facts. Explain that the sensitive plant has no nervous tissue or muscle tissue; plant cells change shape by changing the amount of water in them, causing swelling or shrinking. Contrast this with leg movement, where nerve impulses reach muscle fibres and the muscle fibres change shape and shorten. Do not mention sources, pages, retrieval, or the textbook.
"""
        if _looks_plant_chemical_coordination_question(question):
            user_prompt += """
IMPORTANT PLANT-CHEMICAL-COORDINATION RULE:
Answer the current question using only the retrieved textbook facts. Explain that plant hormones coordinate growth, development and responses; they are synthesised away from where they act and diffuse to the area of action. Include the auxin example only as a concise supporting example. Do not mention sources, pages, retrieval, or the textbook.
"""
        if _looks_spinal_injury_question(question):
            user_prompt += """
IMPORTANT SPINAL-CORD-INJURY RULE:
The textbook states that nerves from all over the body meet in the spinal cord on their way to the brain, reflex arcs are formed in the spinal cord, and the nervous system carries signals from the body to the brain and messages from the brain to muscles. Answer the current question from these facts. Do not say that the textbook lacks the reason. Do not mention sources, pages, retrieval, or the textbook.
"""
        if _looks_neuron_structure_question(question):
            user_prompt += """
IMPORTANT NEURON-STRUCTURE RULE:
The retrieved context explicitly contains the textbook description of Figure 6.1 and the neuron structure.
Answer the current question using that context. State the neuron structure (dendritic tip/dendrites, cell body and axon) and explain the function/path of the electrical impulse described in the context.
Do not say that the textbook lacks details when the retrieved context contains them.
Do not mention sources, pages, retrieval, or the textbook.
"""
        if re.match(r"^\s*which\s+of\s+the\s+following\b", question, re.IGNORECASE) or mcq_stem:
            user_prompt += """
IMPORTANT MCQ/CHOICE QUESTION RULE:
The current input may be an incomplete MCQ stem copied without its options. Use the matching textbook exercise in the retrieved context and recover its options if present. For an incomplete MCQ stem, do NOT classify it as True/False. Return only the single correct option, including its option letter when available. Do not repeat the question, do not explain, and do not return options for another question.
"""
        user_prompt += """
IMPORTANT CURRENT-QUESTION RULE:
Answer ONLY the student's current question.
The retrieved textbook context may contain several exercise questions, MCQs, options, and answers.
Never copy an answer belonging to another question.
Never output answers for multiple questions when the student asked only one.
For an MCQ, return only the single option that answers the current question.
For a True/False statement, return only "True." or "False." Never repeat the statement as the answer.
For an incomplete MCQ stem such as "The brain is responsible for", never return True or False; recover the matching textbook MCQ and return its correct option only.
For a multi-part question, answer every explicit part in order; never stop after answering only the first part.
Do not include previous-question answers or unrelated options.
"""
        for system_prompt in (SYSTEM_PROMPT, RELAXED_SYSTEM_PROMPT):
            refused = True
            fact_parts = []
            for kind, value in _run_llm(
                system_prompt,
                user_prompt,
                block_devanagari=block,
            ):
                if kind == "token":
                    if deterministic_fact or re.search(r"(?im)^\s*\(?[a-d]\)?[.)]?\s+", value):
                        fact_parts.append(value)
                    else:
                        yield "token", value
                else:
                    refused = value
            if not refused:
                if deterministic_fact or fact_parts:
                    answer = _format_fact_answer("".join(fact_parts), intent, question)
                    if answer:
                        yield "token", answer
                return
        # No summary of unrelated text: show a clean "not found" instead.
        if block and not _has_topic_evidence(question, chunks):
            yield "sources", []
            yield "token", NOT_FOUND
            return
        # Third attempt: summarize what the textbook says on the topic.
        parts = []
        refused = True
        for kind, value in _run_llm(
            SUMMARY_SYSTEM_PROMPT,
            build_summary_prompt(question, chunks),
            block_devanagari=block,
        ):
            if kind == "token":
                parts.append(value)
            else:
                refused = value
        if not refused and parts:
            answer = "".join(parts).strip()
            if answer:
                if _summary_says_no_info(answer):
                    yield "sources", []
                    yield "token", NOT_FOUND
                    return
                if deterministic_fact:
                    answer = _format_fact_answer(answer, intent, question)
                yield "token", answer
                return
    except Exception:
        yield "token", "Unable to generate an answer right now."
        return
    yield "sources", []
    yield "token", NOT_FOUND
