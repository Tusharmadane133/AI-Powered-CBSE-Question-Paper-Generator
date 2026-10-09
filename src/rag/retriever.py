import math
import re
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
import config
from src.rag.embeddings import embed_query
_STOP = {
    "what", "which", "that", "this", "with", "from", "have", "does", "about",
    "between", "explain", "describe", "difference", "mean", "means", "kind",
    "give", "name", "write", "their", "there", "them", "they", "your", "into",
    "also", "each", "more", "such", "define", "state", "list", "how", "why",
    "when", "where", "who", "are", "and", "the", "title", "chapter", "grade",
    "science", "textbook",
}
# ============================================================
# BASIC HELPERS
# ============================================================
@lru_cache(maxsize=1)
def _client():
    return QdrantClient(path=config.QDRANT_PATH)
@lru_cache(maxsize=1)
def list_subjects():
    """Return {class_no: [subjects]} from the index."""
    c = _client()
    if not c.collection_exists(config.COLLECTION):
        return {}
    out = {}
    offset = None
    while True:
        pts, offset = c.scroll(
            config.COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["class", "subject"],
            with_vectors=False,
        )
        for p in pts:
            payload = p.payload or {}
            cl = payload.get("class")
            sub = payload.get("subject")
            if cl is None or not sub:
                continue
            out.setdefault(int(cl), set()).add(sub)
        if offset is None:
            break
    return {str(k): sorted(v) for k, v in sorted(out.items())}
def _keywords(text):
    return {
        w
        for w in re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
        if w not in _STOP
    }
def _corpus():
    """
    Load all indexed payloads once.
    Returns:
        payloads -> original Qdrant payloads
        lower    -> lowercase text for lexical searching
    """
    c = _client()
    if not c.collection_exists(config.COLLECTION):
        return [], []
    payloads = []
    lower = []
    offset = None
    while True:
        pts, offset = c.scroll(
            config.COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in pts:
            payload = p.payload or {}
            payloads.append(payload)
            lower.append((payload.get("text") or "").lower())
        if offset is None:
            break
    return payloads, lower
def _key(it):
    return (
        it.get("book"),
        it.get("page"),
        (it.get("text") or "")[:80],
    )
def _normalize_book_name(value):
    """Normalize textbook/chapter names for reliable comparisons."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
def _normalize_class_filter(class_no):
    """Normalize UI class selection. All classes means no class restriction."""
    if class_no is None:
        return None
    value = str(class_no).strip().lower()
    if value in {"", "all", "all classes", "all class", "none", "null"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
def _normalize_subject_filter(subject):
    """Normalize UI subject selection. All subjects means no subject restriction."""
    if subject is None:
        return None
    value = str(subject).strip().lower()
    if value in {"", "all", "all subjects", "all subject", "all-subjects", "none", "null"}:
        return None
    return str(subject).strip()
def _class_ok(payload, class_no):
    class_filter = _normalize_class_filter(class_no)
    if class_filter is None:
        return True
    try:
        return int(payload.get("class", -1)) == class_filter
    except (TypeError, ValueError):
        return False
# ============================================================
# BOOK MATCHING (Curiosity has per-chapter documents)
# ============================================================
@lru_cache(maxsize=16)
def _curiosity_chapter_books(class_no=None, subject=None):
    """
    Discover the individual chapter-document names that belong to
    the Grade 8 Science textbook 'Curiosity'.
    The index stores the parent Curiosity contents separately from
    individual chapter documents, so an exact book == Curiosity filter
    would incorrectly remove the actual chapter documents.
    """
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    payloads, lower = _corpus()
    titles = set()
    for payload, text in zip(payloads, lower):
        if _normalize_book_name(payload.get("book")) != "curiosity":
            continue
        if not _class_ok(payload, class_no):
            continue
        if subject and payload.get("subject") != subject:
            continue
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        for i, line in enumerate(lines):
            match = re.match(
                r"^chapter\s+\d{1,2}\s*[:.)**\\-**–—]?\s*(.*)$",
                line,
                re.IGNORECASE,
            )
            if match:
                title = match.group(1).strip(" .:-–—")
                if len(title) >= 3:
                    titles.add(_normalize_book_name(title))
                elif i + 1 < len(lines):
                    next_line = lines[i + 1].strip(" .:-–—")
                    if 3 <= len(next_line) <= 150:
                        titles.add(_normalize_book_name(next_line))
            # Plain numbered contents entry, e.g.:
            # 6. Pressure, Winds, Storms, and Cyclones
            numbered = re.match(r"^(\d{1,2})\s*[.)**\\-**–—:]\s*(.+)$", line)
            if numbered:
                title = numbered.group(2).strip(" .:-–—")
                if 3 <= len(title) <= 150:
                    titles.add(_normalize_book_name(title))
    return frozenset(t for t in titles if t)
def _book_matches(payload_book, selected_book, class_no=None, subject=None):
    """
    True when an indexed payload belongs to the selected textbook.
    Curiosity is special: its chapter files are indexed under their
    individual chapter titles rather than the parent book name.
    """
    if not selected_book:
        return True
    selected_norm = _normalize_book_name(selected_book)
    payload_norm = _normalize_book_name(payload_book)
    if payload_norm == selected_norm:
        return True
    if selected_norm == "curiosity":
        return payload_norm in _curiosity_chapter_books(class_no, subject)
    return False
# ============================================================
# CHAPTER LOOKUP
# ============================================================
def _extract_chapter_number(question):
    """
    Detect questions such as:
        What is the title of Chapter 6?
        Chapter 6 title?
        What is chapter 10 in Curiosity?
    Returns the chapter number (int) or None.
    """
    q = (question or "").lower().strip()
    patterns = [
        r"\bchapter\s+(\d{1,2})\b",
        r"\bchap\s+(\d{1,2})\b",
        r"\bchapter[-\s]+(\d{1,2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))
    return None
def _is_chapter_title_question(question):
    """Is the user asking for the title/name of a chapter?"""
    q = (question or "").lower()
    if _extract_chapter_number(q) is None:
        return False
    title_words = ("title", "name", "called", "chapter name")
    return any(word in q for word in title_words)
def _chapter_lookup(question, class_no=None, subject=None, book=None, n=5):
    """
    Deterministic chapter-title lookup.
    If the question mentions a textbook such as Curiosity, unrelated
    Prelims/Contents chunks from other books are excluded. This runs
    before vector search because a chapter-title question is a
    structured lookup, not a semantic-similarity problem.
    """
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    chapter_no = _extract_chapter_number(question)
    if chapter_no is None:
        return []
    payloads, lower = _corpus()
    if not payloads:
        return []
    q = (question or "").lower()
    requested_book = (
        _normalize_book_name(book)
        if book
        else ("curiosity" if "curiosity" in q else None)
    )
    candidates = []
    for payload, text in zip(payloads, lower):
        if not _class_ok(payload, class_no):
            continue
        if subject and payload.get("subject") != subject:
            continue
        payload_book = str(payload.get("book") or "").lower()
        exact_chapter = re.search(rf"\bchapter\s+{chapter_no}\b", text)
        numbered_heading = re.search(
                rf"(?m)^\s*{chapter_no}\s*[.):\\-–—]\s*[A-Za-z]",
            rf"(?m)^\s*{chapter_no}\s*[.):\\-–—]\s*[A-Za-z]",
            text,
        )
        numbered_heading_plain = re.search(
                rf"(?m)^\s*{chapter_no}\s+[A-Za-z]",
            rf"(?m)^\s*{chapter_no}\s+[A-Za-z]",
            text,
        )
        if not exact_chapter and not numbered_heading and not numbered_heading_plain:
            continue
        score = 0.0
        if exact_chapter:
            score += 20.0
        if numbered_heading:
            score += 15.0
        if numbered_heading_plain:
            score += 10.0
        if requested_book == "curiosity":
            if payload_book == "curiosity":
                score += 40.0
            if "prelim" in payload_book:
                curiosity_markers = (
                    "curiosity",
                    "textbook of science for grade 8",
                    "exploring the investigative world of science",
                )
                if not any(marker in text for marker in curiosity_markers):
                    continue
                score += 5.0
        chapter_count = len(re.findall(r"\bchapter\s+\d{1,2}\b", text))
        if chapter_count >= 3:
            score += 5.0
        if chapter_count >= 6:
            score += 5.0
        if "prelim" in payload_book:
            score += 3.0
        candidates.append((score, payload))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [payload for _, payload in candidates[:n]]
def _is_chapter_content_question(question):
    """
    Questions asking about the content of a specific chapter
    (main idea, summary, topics, what we learn ...).
    """
    if _extract_chapter_number(question) is None:
        return False
    if _is_chapter_title_question(question):
        return False
    q = (question or "").lower()
    content_words = (
        "main idea",
        "summary",
        "summarize",
        "about",
        "explain",
        "describe",
        "topics",
        "concepts",
        "learn",
        "teach",
        "discuss",
        "important",
        "key points",
        "overview",
    )
    return any(word in q for word in content_words)
def _extract_chapter_title(question, class_no=None, subject=None, book=None):
    """
    Find the requested chapter title from the indexed Contents/Prelims
    chunks. The title is extracted from the source text, not hardcoded.
    """
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    chapter_no = _extract_chapter_number(question)
    if chapter_no is None:
        return None
    payloads, lower = _corpus()
    if not payloads:
        return None
    q = (question or "").lower()
    requested_book = (
        _normalize_book_name(book)
        if book
        else ("curiosity" if "curiosity" in q else None)
    )
    chapter_re = re.compile(rf"\bchapter\s+{chapter_no}\b", re.IGNORECASE)
    next_chapter_re = re.compile(rf"\bchapter\s+{chapter_no + 1}\b", re.IGNORECASE)
    best = None
    for payload, text in zip(payloads, lower):
        if not _class_ok(payload, class_no):
            continue
        if subject and payload.get("subject") != subject:
            continue
        payload_book = str(payload.get("book") or "").lower()
        if requested_book == "curiosity" and _normalize_book_name(payload_book) != "curiosity":
            continue
        match = chapter_re.search(text)
        if not match:
            continue
        after = text[match.end():]
        next_match = next_chapter_re.search(after)
        segment = after[: next_match.start()] if next_match else after[:250]
        segment = re.sub(r"^[\s:.***\\-**–—]+", "", segment)
        segment = re.sub(r"\s+", " ", segment).strip()
        segment = re.split(
            r"\b(?:foreword|about the book|contents|note to|page)\b",
            segment,
            maxsplit=1,
        )[0].strip()
        if not segment:
            continue
        score = 0
        if payload_book == "curiosity":
            score += 100
        if "chapter" in text:
            score += 10
        if len(segment) > 4:
            score += 5
        candidate = (score, segment, payload)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best:
        return best[1]
    return None
def _chapter_content_lookup(question, class_no=None, subject=None, book=None, n=8):
    """
    Resolve the requested chapter title and retrieve the most relevant
    chunks from that exact chapter document.
    """
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    title = _extract_chapter_title(
        question,
        class_no=class_no,
        subject=subject,
        book=book,
    )
    if not title:
        return []
    c = _client()
    if not c.collection_exists(config.COLLECTION):
        return []
    normalized_title = _normalize_book_name(title)
    must = [FieldCondition(key="book", match=MatchValue(value=title))]
    if class_no:
        must.append(FieldCondition(key="class", match=MatchValue(value=int(class_no))))
    if subject:
        must.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
    query_text = f"{question} Chapter {title}"
    try:
        points = c.query_points(
            config.COLLECTION,
            query=embed_query(query_text).tolist(),
            query_filter=Filter(must=must),
            limit=max(config.CANDIDATES, n * 3),
            with_payload=True,
        ).points
    except Exception:
        points = []
    if points:
        results = []
        for point in points:
            payload = point.payload or {}
            vector_score = float(point.score)
            results.append({
                "score": vector_score,
                "vector_score": vector_score,
                "keyword_hits": 0,
                **payload,
            })
        return results[:n]
    # Fallback: match the chapter document by (normalized) name from the corpus.
    payloads, _ = _corpus()
    candidates = [
        payload
        for payload in payloads
        if _normalize_book_name(payload.get("book")) == normalized_title
    ]
    return candidates[:n]
# ============================================================
# EXACT RARE-WORD SEARCH
# ============================================================
def _keyword_search(question, class_no, subject, book=None, n=5):
    """Exact lexical retrieval for meaningful question terms."""
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    payloads, lower = _corpus()
    if not payloads:
        return []
    kws = _keywords(question)
    if not kws:
        return []
    total = len(payloads)
    df = {k: sum(1 for text in lower if re.search(rf"\b{re.escape(k)}\b", text)) for k in kws}
    strong = {k for k in kws if df[k] > 0 and (df[k] <= max(20, total * 0.05) or len(k) >= 8)}
    if not strong:
        strong = {k for k in kws if df[k] > 0}
    scored = []
    for p, text in zip(payloads, lower):
        if not _class_ok(p, class_no):
            continue
        if subject and p.get("subject") != subject:
            continue
        if not _book_matches(p.get("book"), book, class_no=class_no, subject=subject):
            continue
        exact_hits = [k for k in strong if re.search(rf"\b{re.escape(k)}\b", text)]
        if not exact_hits:
            continue
        score = 0.0
        for k in exact_hits:
            score += math.log((total + 1) / (df[k] + 1)) + 1
        scored.append((score, len(exact_hits), p))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [p for _, _, p in scored[:n]]
# ============================================================
# VECTOR SEARCH
# ============================================================
def _vector_items(c, question, class_no, subject, book, top_k):
    """Vector search; returns scored items that belong to the selected book."""
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    must = []
    if class_no:
        must.append(FieldCondition(key="class", match=MatchValue(value=int(class_no))))
    if subject:
        must.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
    res = c.query_points(
        config.COLLECTION,
        query=embed_query(question).tolist(),
        query_filter=Filter(must=must) if must else None,
        limit=max(config.CANDIDATES * 5, top_k * 5, 100),
        with_payload=True,
    ).points
    items = []
    for r in res:
        payload = r.payload or {}
        if not _book_matches(payload.get("book"), book, class_no=class_no, subject=subject):
            continue
        vector_score = float(r.score)
        items.append({
            "score": vector_score,
            "vector_score": vector_score,
            "keyword_hits": 0,
            **payload,
        })
    return items
# ============================================================
# ============================================================
# QUESTION-INTENT / FACT RETRIEVAL
# ============================================================
def _question_intent(question):
    """Classify short factual questions before semantic retrieval."""
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    if re.search(r"\b(si\s+unit|unit\s+of|unit\s+for|what\s+is\s+the\s+unit)\b", q):
        return "unit"
    if re.search(r"\b(why|how\s+does|how\s+do|how\s+is|how\s+are|reason)\b", q):
        return "why"
    if re.search(r"\b(formula|equation|expression|calculate)\b", q):
        return "formula"
    if re.search(r"\b(difference|differentiate|compare|distinguish)\b", q):
        return "comparison"
    if re.search(r"\b(what\s+is|what\s+are|define|definition|meaning\s+of|explain)\b", q):
        return "definition"
    return "general"
def _topic_terms(question):
    """Extract meaningful English topic words from a short factual question."""
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    patterns = [
        r"\bsi\s+unit\s+of\s+(.+?)[?.!]*$",
        r"\bunit\s+of\s+(.+?)[?.!]*$",
        r"\bwhat\s+is\s+the\s+unit\s+of\s+(.+?)[?.!]*$",
        r"\bwhat\s+is\s+(.+?)[?.!]*$",
        r"\bwhat\s+are\s+(.+?)[?.!]*$",
        r"\bdefine\s+(.+?)[?.!]*$",
        r"\bdefinition\s+of\s+(.+?)[?.!]*$",
        r"\bmeaning\s+of\s+(.+?)[?.!]*$",
        r"\bwhy\s+(?:does|do|is|are)\s+(.+?)[?.!]*$",
        r"\bhow\s+(?:does|do|is|are)\s+(.+?)[?.!]*$",
    ]
    topic = None
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            topic = match.group(1).strip()
            break
    if not topic:
        topic = q
    words = re.findall(r"[a-zA-Z]{3,}", topic)
    return [
        w for w in words
        if w not in _STOP and w not in {
            "unit", "units", "what", "does", "give", "explain",
            "define", "definition", "meaning", "reason", "formula",
            "equation", "calculate",
        }
    ]
def _topic_phrase_pattern(question):
    terms = _topic_terms(question)
    if not terms:
        return None
    phrase = r"\s+".join(re.escape(t) for t in terms)
    return rf"\b{phrase}\b"
def _fact_lookup(question, class_no, subject, book=None, n=8):
    """
    Direct textbook retrieval for definition, SI-unit and formula questions.
    First prefer chunks that contain both the topic and explicit fact markers.
    This prevents questions such as "What is electric current?" from being
    routed to a related passage such as "magnetic effect of electric current".
    """
    intent = _question_intent(question)
    if intent not in {"definition", "unit", "formula"}:
        return []
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    payloads, lower = _corpus()
    pattern = _topic_phrase_pattern(question)
    if not payloads or not pattern:
        return []
    candidates = []
    for payload, text in zip(payloads, lower):
        if not _class_ok(payload, class_no):
            continue
        if subject and payload.get("subject") != subject:
            continue
        if not _book_matches(
            payload.get("book"),
            book,
            class_no=class_no,
            subject=subject,
        ):
            continue
        normalized = re.sub(r"\s+", " ", text)
        matches = list(re.finditer(pattern, normalized, re.IGNORECASE))
        if not matches:
            continue
        score = 3.0
        if intent == "definition":
            markers = [
                rf"{pattern}\s+(?:is|are|refers\s+to|means|can\s+be\s+defined\s+as)",
                r"\b(?:defined\s+as|refers\s+to|means|is\s+expressed\s+as)\b",
            ]
            if any(re.search(p, normalized, re.IGNORECASE) for p in markers):
                score += 5.0
            for match in matches:
                window = normalized[
                    max(0, match.start() - 80):match.end() + 220
                ]
                if re.search(
                    r"\b(is|are|refers|means|defined|expressed)\b",
                    window,
                    re.IGNORECASE,
                ):
                    score += 2.0
                    break
        elif intent == "unit":
            marker_hits = sum(
                1
                for marker in (
                    r"\bsi\s+unit\b",
                    r"\bunit\b",
                    r"\bmeasured\s+in\b",
                    r"\bexpressed\s+in\b",
                )
                if re.search(marker, normalized, re.IGNORECASE)
            )
            if marker_hits == 0:
                continue
            score += 2.5 + min(1.5, 0.5 * marker_hits)
        elif intent == "formula":
            marker_hits = sum(
                1
                for marker in (
                    r"\bequation\b",
                    r"\bformula\b",
                    r"\bproportional\b",
                    r"=",
                )
                if re.search(marker, normalized, re.IGNORECASE)
            )
            if marker_hits == 0:
                continue
            score += 2.0 + min(1.5, 0.5 * marker_hits)
        score -= min(len(normalized) / 5000.0, 0.8)
        candidates.append((score, payload))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [payload for _, payload in candidates[:n]]
def _topic_search(question, class_no, subject, book=None, n=8):
    """
    Retrieve chunks containing the exact topic even when the requested fact
    marker is not in the same chunk.
    This prevents a failed short-fact lookup from falling through to unrelated
    vector chunks, especially when the UI is set to All classes / All subjects.
    """
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    payloads, lower = _corpus()
    pattern = _topic_phrase_pattern(question)
    if not payloads or not pattern:
        return []
    intent = _question_intent(question)
    candidates = []
    for payload, text in zip(payloads, lower):
        if not _class_ok(payload, class_no):
            continue
        if subject and payload.get("subject") != subject:
            continue
        if not _book_matches(
            payload.get("book"),
            book,
            class_no=class_no,
            subject=subject,
        ):
            continue
        normalized = re.sub(r"\s+", " ", text)
        matches = list(re.finditer(pattern, normalized, re.IGNORECASE))
        if not matches:
            continue
        score = 1.0 + min(len(matches), 3) * 0.8
        for match in matches:
            window = normalized[
                max(0, match.start() - 120):match.end() + 300
            ]
            if intent == "unit":
                if re.search(
                    r"\b(si\s+unit|unit|measured\s+in|expressed\s+in)\b",
                    window,
                    re.IGNORECASE,
                ):
                    score += 3.0
            elif intent == "definition":
                if re.search(
                    r"\b(is|are|refers\s+to|means|defined|definition)\b",
                    window,
                    re.IGNORECASE,
                ):
                    score += 3.0
            elif intent == "formula":
                if re.search(
                    r"\b(formula|equation|proportional)\b|=",
                    window,
                    re.IGNORECASE,
                ):
                    score += 3.0
        score -= min(len(normalized) / 6000.0, 0.5)
        candidates.append((score, payload))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [payload for _, payload in candidates[:n]]
# NORMAL HYBRID RETRIEVAL
# ============================================================
def _is_incomplete_mcq_stem(question):
    q=str(question or "").strip()
    if not q or "?" in q:
        return False
    ql=q.lower().rstrip(".!,:;").strip()
    if len(ql.split())<4:
        return False
    endings=(
        r"\bresponsible\s+for$",r"\bknown\s+as$",r"\bcalled$",
        r"\breferred\s+to\s+as$",r"\bconsists?\s+of$",
        r"\bused\s+(?:for|to)$",r"\bdefined\s+as$",
        r"\bis\s+(?:a|an|the)$",r"\bare\s+(?:a|an|the)$",
        r"\bis\s+the$",r"\bare\s+the$"
    )
    return any(re.search(pattern,ql) for pattern in endings)
def _mcq_stem_terms(question):
    q=re.sub(r"[^a-zA-Z0-9\s]"," ",str(question or "").lower())
    stop=set(_STOP)|{"is","are","was","were","be","been","being","a","an","the","of","to","in","on","for","and","or"}
    return [w for w in re.findall(r"[a-zA-Z]{3,}",q) if w not in stop]
def _mcq_stem_lookup(question,class_no,subject,book=None,n=8):
    terms=list(dict.fromkeys(_mcq_stem_terms(question)))
    if not terms:
        return []
    class_no=_normalize_class_filter(class_no)
    subject=_normalize_subject_filter(subject)
    payloads,lower=_corpus()
    candidates=[]
    for payload,text in zip(payloads,lower):
        if not _class_ok(payload,class_no):
            continue
        if subject and payload.get("subject")!=subject:
            continue
        if not _book_matches(payload.get("book"),book,class_no=class_no,subject=subject):
            continue
        normalized=re.sub(r"\s+"," ",text)
        hits=sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b",normalized,re.IGNORECASE))
        if hits<max(2,len(terms)-1):
            continue
        option_hits=len(re.findall(r"(?m)\(?[a-d]\)[.)]?\s+",normalized,re.IGNORECASE))
        exact_phrase=" ".join(terms)
        score=hits*2.0
        if exact_phrase and all(term in normalized for term in terms):
            score+=5.0
        if option_hits>=2:
            score+=8.0
        if re.search(r"\b(exercises|exercise)\b",normalized,re.IGNORECASE):
            score+=2.0
        candidates.append((score,payload))
    candidates.sort(key=lambda x:x[0],reverse=True)
    return [payload for _,payload in candidates[:n]]
def _is_fill_blank_question(question):
    q=str(question or "").strip()
    return bool(re.search(r"_{2,}|\.{4,}|…+",q))
def _fill_blank_terms(question):
    q=re.sub(r"_{2,}|\.{4,}|…+"," ",str(question or "").lower())
    words=re.findall(r"[a-zA-Z]{3,}",q)
    stop=set(_STOP)|{"is","are","was","were","be","been","being","the","a","an","of","to","in","on","for","and","or"}
    return [w for w in words if w not in stop]
def _fill_blank_lookup(question,class_no,subject,book=None,n=8):
    terms=list(dict.fromkeys(_fill_blank_terms(question)))
    if not terms:
        return []
    class_no=_normalize_class_filter(class_no)
    subject=_normalize_subject_filter(subject)
    payloads,lower=_corpus()
    candidates=[]
    for payload,text in zip(payloads,lower):
        if not _class_ok(payload,class_no):
            continue
        if subject and payload.get("subject")!=subject:
            continue
        if not _book_matches(payload.get("book"),book,class_no=class_no,subject=subject):
            continue
        positions={}
        for term in terms:
            match=re.search(rf"\b{re.escape(term)}\b",text)
            if match:
                positions[term]=match.start()
        hits=len(positions)
        if hits==0:
            continue
        if len(terms)>=3 and hits<max(2,len(terms)-1):
            continue
        span=0
        if len(positions)>=2:
            values=sorted(positions.values())
            span=values[-1]-values[0]
        coverage=hits/len(terms)
        score=10.0*coverage+max(0.0,3.0-min(span/120.0,3.0))
        if hits==len(terms):
            score+=5.0
        candidates.append((score,hits,payload))
    candidates.sort(key=lambda item:(item[0],item[1]),reverse=True)
    return [payload for _,_,payload in candidates[:n]]
def retrieve(
    question,
    class_no=None,
    subject=None,
    top_k=config.TOP_K,
    book=None,
):
    """
    Hybrid retrieval:
    1. Structured chapter-title lookup
    2. Chapter content lookup
    3. Vector search (+ book filter, with fallback to all books)
    4. Keyword boosting
    5. Exact rare-word retrieval
    """
    # "All" selections from the UI mean no restriction.
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    # When both class and subject are All, ignore any stale book value sent by the UI.
    # Otherwise an old book such as "Exploring Forces" can block the correct book.
    if class_no is None and subject is None:
        book = None
    if not book and "curiosity" in (question or "").lower():
        book = "Curiosity"
    c = _client()
    if not c.collection_exists(config.COLLECTION):
        return []
    # STEP 1 - STRUCTURED CHAPTER LOOKUP
    if _is_chapter_title_question(question):
        chapter_results = _chapter_lookup(
            question,
            class_no=class_no,
            subject=subject,
            book=book,
            n=max(5, top_k),
        )
        if chapter_results:
            results = []
            for index, payload in enumerate(chapter_results):
                results.append({
                    "score": 1.20 - (index * 0.01),   # deterministic matches rank first
                    "vector_score": 0.0,
                    "keyword_hits": 1,
                    **payload,
                })
            return results[:top_k]
    # STEP 2 - CHAPTER CONTENT LOOKUP
    if _is_chapter_content_question(question):
        content_results = _chapter_content_lookup(
            question,
            class_no=class_no,
            subject=subject,
            book=book,
            n=max(8, top_k),
        )
        if content_results:
            results = []
            for index, payload in enumerate(content_results):
                results.append({
                    "score": 1.19 - (index * 0.01),
                    "vector_score": 0.0,
                    "keyword_hits": 1,
                    **payload,
                })
            return results[:top_k]
    # STEP 3 - DETERMINISTIC INCOMPLETE-MCQ LOOKUP
    if _is_incomplete_mcq_stem(question):
        mcq_results=_mcq_stem_lookup(question,class_no=class_no,subject=subject,book=book,n=max(8,top_k))
        if mcq_results:
            results=[]
            for index,payload in enumerate(mcq_results):
                results.append({
                    "score":1.70-(index*0.01),
                    "vector_score":0.0,
                    "keyword_hits":len(_mcq_stem_terms(question)),
                    **payload,
                })
            return results[:top_k]
    # STEP 3 - DETERMINISTIC FILL-IN-THE-BLANK LOOKUP
    if _is_fill_blank_question(question):
        fill_results=_fill_blank_lookup(question,class_no=class_no,subject=subject,book=book,n=max(8,top_k))
        if not fill_results:
            fill_terms=list(dict.fromkeys(_fill_blank_terms(question)))
            topic_results=_topic_search(question,class_no=class_no,subject=subject,book=book,n=max(12,top_k))
            strict=[]
            for payload in topic_results:
                text=str(payload.get("text") or "").lower()
                hits=sum(1 for term in fill_terms if re.search(rf"\b{re.escape(term)}\b",text))
                if hits>=max(2,min(3,len(fill_terms))):
                    strict.append(payload)
            fill_results=strict
        if fill_results:
            results=[]
            for index,payload in enumerate(fill_results):
                results.append({
                    "score":1.60-(index*0.01),
                    "vector_score":0.0,
                    "keyword_hits":len(_fill_blank_terms(question)),
                    **payload,
                })
            return results[:top_k]
    # STEP 4 - DETERMINISTIC SHORT-FACT LOOKUP
    # Definition/unit/formula questions first look for direct textbook evidence.
    fact_results = _fact_lookup(
        question,
        class_no=class_no,
        subject=subject,
        book=book,
        n=max(8, top_k),
    )
    if fact_results:
        results = []
        for index, payload in enumerate(fact_results):
            results.append({
                "score": 1.50 - (index * 0.01),
                "vector_score": 0.0,
                "keyword_hits": 2,
                **payload,
            })
        return results[:top_k]
    # If an exact fact marker is not present in the same chunk, keep retrieval
    # anchored to the exact topic before trying semantic/vector search.
    intent = _question_intent(question)
    if intent in {"definition", "unit", "formula"}:
        topic_results = _topic_search(
            question,
            class_no=class_no,
            subject=subject,
            book=book,
            n=max(8, top_k),
        )
        if topic_results:
            results = []
            for index, payload in enumerate(topic_results):
                results.append({
                    "score": 1.30 - (index * 0.01),
                    "vector_score": 0.0,
                    "keyword_hits": 1,
                    **payload,
                })
            return results[:top_k]
    # STEP 4 - VECTOR RETRIEVAL
    items = _vector_items(c, question, class_no, subject, book, top_k)
    if not items and book:
        # The book filter matched nothing (name mismatch): search all books instead.
        book = None
        items = _vector_items(c, question, class_no, subject, None, top_k)
    # STEP 5 - KEYWORD BOOST
    kws = _keywords(question)
    query_terms = {
        w
        for w in re.findall(r"[a-zA-Z]{3,}", (question or "").lower())
        if w not in _STOP
    }
    for it in items:
        text = (it.get("text") or "").lower()
        hits = sum(1 for k in kws if re.search(rf"\b{re.escape(k)}\b", text))
        term_hits = sum(
            1
            for k in query_terms
            if re.search(rf"\b{re.escape(k)}\b", text)
        )
        it["keyword_hits"] = max(hits, term_hits)
        lexical_boost = min(config.KEYWORD_BOOST * hits, config.KEYWORD_BOOST_MAX)
        if term_hits >= 2:
            lexical_boost += min(0.08, 0.02 * term_hits)
        it["score"] = it["vector_score"] + lexical_boost
    # STEP 6 - EXACT RARE-WORD SEARCH
    by_key = {_key(it): it for it in items}
    for p in _keyword_search(question, class_no, subject, book=book):
        base = getattr(config, "KEYWORD_SCORE_BASE", 0.85)
        k = _key(p)
        if k in by_key:
            by_key[k]["score"] = max(by_key[k]["score"], base)
        else:
            new = {
                "score": base,
                "vector_score": 0.0,
                "keyword_hits": 1,
                **p,
            }
            items.append(new)
            by_key[k] = new
    # STEP 7 - TOPIC RELEVANCE GATE
    # Prefer exact textbook evidence for meaningful terms. If the index has no
    # lexical evidence for the question, do not send unrelated vector chunks
    # to the LLM because they can cause an incorrect answer from another topic.
    exact_payloads = _keyword_search(question, class_no, subject, book=book, n=max(top_k * 3, 10))
    exact_keys = {_key(p) for p in exact_payloads}
    if exact_keys:
        for it in items:
            if _key(it) in exact_keys:
                it["score"] = max(float(it.get("score", 0.0)), 2.0)
        items = [it for it in items if _key(it) in exact_keys or it.get("keyword_hits", 0) >= 2]
    elif not any(it.get("keyword_hits", 0) > 0 for it in items):
        return []
    # STEP 8 - FINAL SORT
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:top_k]
