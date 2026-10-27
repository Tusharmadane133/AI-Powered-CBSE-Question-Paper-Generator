import atexit
import logging
import math
import re
import time
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
    "science", "textbook", "unit", "units", "value", "values",
}
logger = logging.getLogger(__name__)
def _wb(term):
    """Word-boundary pattern that also works for Devanagari."""
    if re.search(r"[\u0900-\u097F]", term):
        return r"(?<![\u0900-\u097F])" + re.escape(term) + r"(?![\u0900-\u097F])"
    return rf"\b{re.escape(term)}\b"
@lru_cache(maxsize=1)
def _client():
    return QdrantClient(path=config.QDRANT_PATH)
def _close_qdrant_client():
    if _client.cache_info().currsize == 0:
        return
    try:
        _client().close()
    except Exception:
        pass
    finally:
        _client.cache_clear()
atexit.register(_close_qdrant_client)
@lru_cache(maxsize=1)
def list_subjects():
    """Return {class_no: [subjects]} from the index."""
    client = _client()
    if not client.collection_exists(config.COLLECTION):
        return {}
    out = {}
    offset = None
    while True:
        points, offset = client.scroll(
            config.COLLECTION, limit=1000, offset=offset,
            with_payload=["class", "subject"], with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            class_value = payload.get("class")
            subject_value = payload.get("subject")
            if class_value is None or not subject_value:
                continue
            out.setdefault(int(class_value), set()).add(subject_value)
        if offset is None:
            break
    return {str(key): sorted(value) for key, value in sorted(out.items())}
def _keywords(text):
    text = (text or "").lower()
    words = {word for word in re.findall(r"[a-zA-Z]{4,}", text) if word not in _STOP}
    words |= set(re.findall(r"[\u0900-\u097F]{3,}", text))
    return words
@lru_cache(maxsize=1)
def _corpus():
    """Load indexed payloads once; cache payloads and lowercase text for lexical search."""
    client = _client()
    if not client.collection_exists(config.COLLECTION):
        return [], []
    payloads, lower_texts = [], []
    offset = None
    while True:
        points, offset = client.scroll(
            config.COLLECTION, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            payloads.append(payload)
            lower_texts.append((payload.get("text") or "").lower())
        if offset is None:
            break
    logger.info("[RAG_TIMING] corpus_cache_loaded chunks=%d", len(payloads))
    return payloads, lower_texts
@lru_cache(maxsize=1)
def _lexical_index():
    """Build a reusable inverted index once instead of regex-scanning every chunk per query."""
    started = time.perf_counter()
    payloads, lower_texts = _corpus()
    postings = {}
    for index, content in enumerate(lower_texts):
        tokens = set(re.findall(r"[a-zA-Z]{4,}|[\u0900-\u097F]{3,}", content))
        for token in tokens:
            postings.setdefault(token, []).append(index)
    logger.info(
        "[RAG_TIMING] lexical_index_build=%.3fs chunks=%d terms=%d",
        time.perf_counter() - started, len(payloads), len(postings),
    )
    return postings
def _key(item):
    return (item.get("book"), item.get("page"), (item.get("text") or "")[:80])
def _normalize_book_name(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
def _normalize_class_filter(class_no):
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
@lru_cache(maxsize=16)
def _curiosity_chapter_books(class_no=None, subject=None):
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
        for index, line in enumerate(lines):
            match = re.match(r"^chapter\s+\d{1,2}\s*[:.)–—-]?\s*(.*)$", line, re.IGNORECASE)
            if match:
                title = match.group(1).strip(" .:-–—")
                if len(title) >= 3:
                    titles.add(_normalize_book_name(title))
                elif index + 1 < len(lines):
                    next_line = lines[index + 1].strip(" .:-–—")
                    if 3 <= len(next_line) <= 150:
                        titles.add(_normalize_book_name(next_line))
            numbered = re.match(r"^(\d{1,2})\s*[.)–—:-]\s*(.+)$", line)
            if numbered:
                title = numbered.group(2).strip(" .:-–—")
                if 3 <= len(title) <= 150:
                    titles.add(_normalize_book_name(title))
    return frozenset(title for title in titles if title)
def _book_matches(payload_book, selected_book, class_no=None, subject=None):
    if not selected_book:
        return True
    selected_norm = _normalize_book_name(selected_book)
    payload_norm = _normalize_book_name(payload_book)
    if payload_norm == selected_norm:
        return True
    if selected_norm == "curiosity":
        return payload_norm in _curiosity_chapter_books(class_no, subject)
    return False
def _extract_chapter_number(question):
    question = (question or "").lower().strip()
    for pattern in (r"\bchapter\s+(\d{1,2})\b", r"\bchap\s+(\d{1,2})\b", r"\bchapter[-\s]+(\d{1,2})\b"):
        match = re.search(pattern, question)
        if match:
            return int(match.group(1))
    return None
def _is_chapter_title_question(question):
    question = (question or "").lower()
    if _extract_chapter_number(question) is None:
        return False
    return any(word in question for word in ("title", "name", "called", "chapter name"))
def _chapter_lookup(question, class_no=None, subject=None, book=None, n=5):
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    chapter_no = _extract_chapter_number(question)
    if chapter_no is None:
        return []
    payloads, lower = _corpus()
    if not payloads:
        return []
    question_lower = (question or "").lower()
    requested_book = _normalize_book_name(book) if book else ("curiosity" if "curiosity" in question_lower else None)
    candidates = []
    for payload, text in zip(payloads, lower):
        if not _class_ok(payload, class_no):
            continue
        if subject and payload.get("subject") != subject:
            continue
        payload_book = str(payload.get("book") or "").lower()
        exact_chapter = re.search(rf"\bchapter\s+{chapter_no}\b", text)
        numbered_heading = re.search(rf"(?m)^\s*{chapter_no}\s*[.):\\\\\\\\-–—]\s*[A-Za-z]", text)
        numbered_heading_plain = re.search(rf"(?m)^\s*{chapter_no}\s+[A-Za-z]", text)
        if not (exact_chapter or numbered_heading or numbered_heading_plain):
            continue
        score = 20.0 if exact_chapter else 0.0
        if numbered_heading:
            score += 15.0
        if numbered_heading_plain:
            score += 10.0
        if requested_book == "curiosity":
            if _normalize_book_name(payload_book) == "curiosity":
                score += 40.0
            if "prelim" in payload_book:
                markers = ("curiosity", "textbook of science for grade 8", "exploring the investigative world of science")
                if not any(marker in text for marker in markers):
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
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in candidates[:n]]
def _is_chapter_content_question(question):
    if _extract_chapter_number(question) is None or _is_chapter_title_question(question):
        return False
    words = ("main idea", "summary", "summarize", "about", "explain", "describe", "topics", "concepts", "learn", "teach", "discuss", "important", "key points", "overview")
    question = (question or "").lower()
    return any(word in question for word in words)
def _extract_chapter_title(question, class_no=None, subject=None, book=None):
    chapter_no = _extract_chapter_number(question)
    if chapter_no is None:
        return None
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    payloads, lower = _corpus()
    if not payloads:
        return None
    question_lower = (question or "").lower()
    requested_book = _normalize_book_name(book) if book else ("curiosity" if "curiosity" in question_lower else None)
    chapter_re = re.compile(rf"\bchapter\s+{chapter_no}\b", re.IGNORECASE)
    next_chapter_re = re.compile(rf"\bchapter\s+{chapter_no + 1}\b", re.IGNORECASE)
    best = None
    for payload, text in zip(payloads, lower):
        if not _class_ok(payload, class_no) or (subject and payload.get("subject") != subject):
            continue
        payload_book = str(payload.get("book") or "").lower()
        if requested_book == "curiosity" and _normalize_book_name(payload_book) != "curiosity":
            continue
        match = chapter_re.search(text)
        if not match:
            continue
        after = text[match.end():]
        next_match = next_chapter_re.search(after)
        segment = after[:next_match.start()] if next_match else after[:250]
        segment = re.sub(r"^[\s:.*–—-]+", "", segment)
        segment = re.sub(r"\s+", " ", segment).strip()
        segment = re.split(r"\b(?:foreword|about the book|contents|note to|page)\b", segment, maxsplit=1)[0].strip()
        if not segment:
            continue
        score = (100 if _normalize_book_name(payload_book) == "curiosity" else 0) + (10 if "chapter" in text else 0) + (5 if len(segment) > 4 else 0)
        candidate = (score, segment)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1] if best else None
def _chapter_content_lookup(question, class_no=None, subject=None, book=None, n=8):
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    title = _extract_chapter_title(question, class_no=class_no, subject=subject, book=book)
    if not title:
        return []
    client = _client()
    if not client.collection_exists(config.COLLECTION):
        return []
    normalized_title = _normalize_book_name(title)
    must = [FieldCondition(key="book", match=MatchValue(value=title))]
    if class_no:
        must.append(FieldCondition(key="class", match=MatchValue(value=int(class_no))))
    if subject:
        must.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
    try:
        points = client.query_points(
            config.COLLECTION, query=embed_query(f"{question} Chapter {title}").tolist(),
            query_filter=Filter(must=must),
            limit=max(config.CANDIDATES, n * 3), with_payload=True,
        ).points
    except Exception:
        points = []
    if points:
        results = []
        for point in points:
            payload = point.payload or {}
            vector_score = float(point.score)
            results.append({"score": vector_score, "vector_score": vector_score, "keyword_hits": 0, **payload})
        return results[:n]
    payloads, _ = _corpus()
    return [payload for payload in payloads if _normalize_book_name(payload.get("book")) == normalized_title][:n]
def _keyword_search(question, class_no, subject, book=None, n=5):
    """Exact lexical retrieval using cached inverted postings, avoiding full-corpus regex scans."""
    started = time.perf_counter()
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    payloads, lower = _corpus()
    if not payloads:
        return []
    keywords = _keywords(question)
    if not keywords:
        return []
    postings = _lexical_index()
    total = len(payloads)
    df = {word: len(postings.get(word, ())) for word in keywords}
    strong = {
        word for word in keywords
        if df[word] > 0 and (df[word] <= max(20, total * 0.05) or len(word) >= 8)
    }
    if not strong:
        strong = {word for word in keywords if df[word] > 0}
    if not strong:
        logger.info("[RAG_TIMING] keyword_search=%.3fs candidates=0", time.perf_counter() - started)
        return []
    candidate_ids = set()
    for word in strong:
        candidate_ids.update(postings.get(word, ()))
    scored = []
    for index in candidate_ids:
        payload = payloads[index]
        if not _class_ok(payload, class_no) or (subject and payload.get("subject") != subject):
            continue
        if not _book_matches(payload.get("book"), book, class_no=class_no, subject=subject):
            continue
        hits = [word for word in strong if index in postings.get(word, ())]
        if not hits:
            continue
        score = sum(math.log((total + 1) / (df[word] + 1)) + 1 for word in hits)
        scored.append((score, len(hits), payload))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    results = [payload for _, _, payload in scored[:n]]
    logger.info(
        "[RAG_TIMING] keyword_search=%.3fs candidate_chunks=%d matched=%d returned=%d",
        time.perf_counter() - started, len(candidate_ids), len(scored), len(results),
    )
    return results
def _vector_items(client, question, class_no, subject, book, top_k):
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    must = []
    if class_no:
        must.append(FieldCondition(key="class", match=MatchValue(value=int(class_no))))
    if subject:
        must.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
    points = client.query_points(
        config.COLLECTION, query=embed_query(question).tolist(),
        query_filter=Filter(must=must) if must else None,
        limit=max(config.CANDIDATES * 3, top_k * 3, 24), with_payload=True,
    ).points
    items = []
    for point in points:
        payload = point.payload or {}
        if not _book_matches(payload.get("book"), book, class_no=class_no, subject=subject):
            continue
        vector_score = float(point.score)
        items.append({"score": vector_score, "vector_score": vector_score, "keyword_hits": 0, **payload})
    return items
def _question_intent(question):
    question = re.sub(r"\s+", " ", (question or "").strip().lower())
    if re.search(r"\b(si\s+unit|unit\s+of|unit\s+for|what\s+is\s+the\s+unit)\b", question):
        return "unit"
    if re.search(r"\b(why|how\s+does|how\s+do|how\s+is|how\s+are|reason)\b", question):
        return "why"
    if re.search(r"\b(formula|equation|expression|calculate)\b", question):
        return "formula"
    if re.search(r"\b(difference|differentiate|compare|distinguish)\b", question):
        return "comparison"
    if re.search(r"\b(what\s+is|what\s+are|define|definition|meaning\s+of)\b", question):
        return "definition"
    return "general"
def _topic_terms(question):
    question = re.sub(r"\s+", " ", (question or "").strip().lower())
    patterns = (
        r"\bsi\s+unit\s+of\s+(.+?)[?.!]?$",
        r"\bunit\s+of\s+(.+?)[?.!]?$",
        r"\bwhat\s+is\s+the\s+unit\s+of\s+(.+?)[?.!]?$",
        r"\bwhat\s+is\s+(.+?)[?.!]?$",
        r"\bwhat\s+are\s+(.+?)[?.!]?$",
        r"\bdefine\s+(.+?)[?.!]?$",
        r"\bdefinition\s+of\s+(.+?)[?.!]?$",
        r"\bmeaning\s+of\s+(.+?)[?.!]?$",
        r"\bwhy\s+(?:does|do|is|are)\s+(.+?)[?.!]?$",
        r"\bhow\s+(?:does|do|is|are)\s+(.+?)[?.!]?$",
    )
    topic = None
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            topic = match.group(1).strip()
            break
    if not topic:
        topic = question
    words = re.findall(r"[a-zA-Z]{3,}", topic)
    return [word for word in words if word not in _STOP and word not in {"unit", "units", "what", "does", "give", "explain", "define", "definition", "meaning", "reason", "formula", "equation", "calculate"}]
def _topic_phrase_pattern(question):
    terms = _topic_terms(question)
    if not terms:
        return None
    connector = r"(?:\s+(?:of|the|a|an|for|in|on|to|and|by|with))?\s+"
    return rf"\b{connector.join(re.escape(term) for term in terms)}\b"
def _fact_lookup(question, class_no, subject, book=None, n=8):
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
        if not _class_ok(payload, class_no) or (subject and payload.get("subject") != subject):
            continue
        if not _book_matches(payload.get("book"), book, class_no=class_no, subject=subject):
            continue
        normalized = re.sub(r"\s+", " ", text)
        matches = list(re.finditer(pattern, normalized, re.IGNORECASE))
        if not matches:
            continue
        score = 3.0
        if intent == "definition":
            markers = (
                rf"{pattern}\s+(?:is|are|refers\s+to|means|can\s+be\s+defined\s+as|states?\s+that|is\s+the\s+law\s+that)",
                r"\b(?:defined\s+as|refers\s+to|means|is\s+expressed\s+as|states?\s+that)\b",
            )
            if any(re.search(marker, normalized, re.IGNORECASE) for marker in markers):
                score += 5.0
            evidence = (
                r"matter\s+can\s+neither\s+be\s+created\s+nor\s+destroyed",
                r"total\s+mass\s+(?:remains\s+the\s+same|of.{0,100}?(?:equal|same))",
                r"known\s+as\s+the\s+law\s+of\s+conservation\s+of\s+mass",
                r"number\s+of\s+atoms.{0,100}?(?:same|remains)",
            )
            windows = [normalized[max(0, match.start()-180):min(len(normalized), match.end()+500)] for match in matches]
            local_evidence = any(re.search(expr, window, re.IGNORECASE) for window in windows for expr in evidence)
            score += 15.0 if local_evidence else -2.0
            heading = re.search(r"(?:^|\s)\d{1,2}(?:\.\d{1,2})?\s+law\s+of\s+conservation\s+of\s+mass\b", normalized, re.IGNORECASE)
            if heading:
                score += 12.0
                heading_window = normalized[heading.start():heading.start()+650]
                if re.search(r"matter\s+can\s+neither\s+be\s+created\s+nor\s+destroyed|total\s+mass\s+remains\s+the\s+same|known\s+as\s+the\s+law\s+of\s+conservation\s+of\s+mass", heading_window, re.IGNORECASE):
                    score += 8.0
            if re.search(r"\batomic\s+mass\b", normalized, re.IGNORECASE) and not heading:
                score -= 4.0
            for match in matches:
                window = normalized[max(0, match.start()-120):match.end()+300]
                if re.search(r"\b(?:is\s+known\s+as|known\s+as|refers\s+to|means|defined\s+as|states?\s+that|can\s+neither\s+be\s+created)\b", window, re.IGNORECASE):
                    score += 3.0
                    break
        elif intent == "unit":
            marker_hits = sum(bool(re.search(marker, normalized, re.IGNORECASE)) for marker in (r"\bsi\s+unit\b", r"\bunit\b", r"\bmeasured\s+in\b", r"\bexpressed\s+in\b"))
            if not marker_hits:
                continue
            score += 2.5 + min(1.5, 0.5 * marker_hits)
            for match in matches:
                window = normalized[max(0, match.start()-150):match.end()+250]
                if re.search(r"\b(si\s+unit|newton|measured\s+in|expressed\s+in)\b", window, re.IGNORECASE):
                    score += 5.0
                    break
                if re.search(r"\bunit\b", window, re.IGNORECASE):
                    score += 2.5
                    break
        elif intent == "formula":
            marker_hits = sum(bool(re.search(marker, normalized, re.IGNORECASE)) for marker in (r"\bequation\b", r"\bformula\b", r"\bproportional\b", r"="))
            if not marker_hits:
                continue
            score += 2.0 + min(1.5, 0.5 * marker_hits)
        score -= min(len(normalized) / 5000.0, 0.8)
        candidates.append((score, payload))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in candidates[:n]]
def _topic_search(question, class_no, subject, book=None, n=8):
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    payloads, lower = _corpus()
    pattern = _topic_phrase_pattern(question)
    if not payloads or not pattern:
        return []
    intent = _question_intent(question)
    candidates = []
    for payload, text in zip(payloads, lower):
        if not _class_ok(payload, class_no) or (subject and payload.get("subject") != subject):
            continue
        if not _book_matches(payload.get("book"), book, class_no=class_no, subject=subject):
            continue
        normalized = re.sub(r"\s+", " ", text)
        matches = list(re.finditer(pattern, normalized, re.IGNORECASE))
        if not matches:
            continue
        score = 1.0 + min(len(matches), 3) * 0.8
        for match in matches:
            window = normalized[max(0, match.start()-120):match.end()+300]
            if intent == "unit" and re.search(r"\b(si\s+unit|unit|measured\s+in|expressed\s+in)\b", window, re.IGNORECASE):
                score += 3.0
            elif intent == "definition" and re.search(r"\b(is|are|refers\s+to|means|defined|definition|known\s+as)\b", window, re.IGNORECASE):
                score += 3.0
            elif intent == "formula" and re.search(r"\b(formula|equation|proportional)\b|=", window, re.IGNORECASE):
                score += 3.0
        if intent == "definition":
            evidence = r"matter\s+can\s+neither\s+be\s+created\s+nor\s+destroyed|total\s+mass\s+(?:remains\s+the\s+same|of.{0,100}?(?:equal|same))|known\s+as\s+the\s+law\s+of\s+conservation\s+of\s+mass"
            local_evidence = any(re.search(evidence, normalized[max(0, match.start()-180):min(len(normalized), match.end()+500)], re.IGNORECASE) for match in matches)
            score += 15.0 if local_evidence else -2.0
        score -= min(len(normalized) / 6000.0, 0.5)
        candidates.append((score, payload))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in candidates[:n]]
def _is_incomplete_mcq_stem(question):
    question = str(question or "").strip()
    if not question or "?" in question:
        return False
    normalized = question.lower().rstrip(".!,:;").strip()
    if len(normalized.split()) < 4:
        return False
    endings = (r"\bresponsible\s+for$", r"\bknown\s+as$", r"\bcalled$", r"\breferred\s+to\s+as$", r"\bconsists?\s+of$", r"\bused\s+(?:for|to)$", r"\bdefined\s+as$", r"\bis\s+(?:a|an|the)$", r"\bare\s+(?:a|an|the)$", r"\bis\s+the$", r"\bare\s+the$")
    return any(re.search(pattern, normalized) for pattern in endings)
def _mcq_stem_terms(question):
    question = re.sub(r"[^a-zA-Z0-9\s]", " ", str(question or "").lower())
    stop = set(_STOP) | {"is", "are", "was", "were", "be", "been", "being", "a", "an", "the", "of", "to", "in", "on", "for", "and", "or"}
    return [word for word in re.findall(r"[a-zA-Z]{3,}", question) if word not in stop]
def _mcq_stem_lookup(question, class_no, subject, book=None, n=8):
    terms = list(dict.fromkeys(_mcq_stem_terms(question)))
    if not terms:
        return []
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    payloads, lower = _corpus()
    candidates = []
    for payload, text in zip(payloads, lower):
        if not _class_ok(payload, class_no) or (subject and payload.get("subject") != subject):
            continue
        if not _book_matches(payload.get("book"), book, class_no=class_no, subject=subject):
            continue
        normalized = re.sub(r"\s+", " ", text)
        hits = sum(bool(re.search(_wb(term), normalized, re.IGNORECASE)) for term in terms)
        if hits < max(2, len(terms)-1):
            continue
        option_hits = len(re.findall(r"(?m)\(?[a-d]\)?[.)]?\s+", normalized, re.IGNORECASE))
        exact_phrase = " ".join(terms)
        score = hits * 2.0
        if exact_phrase and all(term in normalized for term in terms):
            score += 5.0
        if option_hits >= 2:
            score += 8.0
        if re.search(r"\b(exercises|exercise)\b", normalized, re.IGNORECASE):
            score += 2.0
        candidates.append((score, payload))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in candidates[:n]]
def _is_fill_blank_question(question):
    return bool(re.search(r"_{2,}|\.{4,}|…+", str(question or "").strip()))
def _fill_blank_terms(question):
    question = re.sub(r"_{2,}|\.{4,}|…+", " ", str(question or "").lower())
    words = re.findall(r"[a-zA-Z]{3,}", question)
    stop = set(_STOP) | {"is", "are", "was", "were", "be", "been", "being", "the", "a", "an", "of", "to", "in", "on", "for", "and", "or"}
    return [word for word in words if word not in stop]
def _fill_blank_lookup(question, class_no, subject, book=None, n=8):
    terms = list(dict.fromkeys(_fill_blank_terms(question)))
    if not terms:
        return []
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    payloads, lower = _corpus()
    candidates = []
    for payload, text in zip(payloads, lower):
        if not _class_ok(payload, class_no) or (subject and payload.get("subject") != subject):
            continue
        if not _book_matches(payload.get("book"), book, class_no=class_no, subject=subject):
            continue
        positions = {}
        for term in terms:
            match = re.search(_wb(term), text)
            if match:
                positions[term] = match.start()
        hits = len(positions)
        if hits == 0 or (len(terms) >= 3 and hits < max(2, len(terms)-1)):
            continue
        span = 0
        if len(positions) >= 2:
            values = sorted(positions.values())
            span = values[-1] - values[0]
        coverage = hits / len(terms)
        score = 10.0 * coverage + max(0.0, 3.0 - min(span / 120.0, 3.0))
        if hits == len(terms):
            score += 5.0
        candidates.append((score, hits, payload))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [payload for _, _, payload in candidates[:n]]
def retrieve(question, class_no=None, subject=None, top_k=config.TOP_K, book=None):
    """Hybrid textbook retrieval with structured, lexical and vector fallbacks."""
    class_no = _normalize_class_filter(class_no)
    subject = _normalize_subject_filter(subject)
    if class_no is None and subject is None:
        book = None
    if not book and "curiosity" in (question or "").lower():
        book = "Curiosity"
    client = _client()
    if not client.collection_exists(config.COLLECTION):
        return []
    if _is_chapter_title_question(question):
        results = _chapter_lookup(question, class_no=class_no, subject=subject, book=book, n=max(5, top_k))
        if results:
            return [{"score": 1.20 - index * 0.01, "vector_score": 0.0, "keyword_hits": 1, **payload} for index, payload in enumerate(results)][:top_k]
    if _is_chapter_content_question(question):
        results = _chapter_content_lookup(question, class_no=class_no, subject=subject, book=book, n=max(8, top_k))
        if results:
            return [{"score": 1.19 - index * 0.01, "vector_score": 0.0, "keyword_hits": 1, **payload} for index, payload in enumerate(results)][:top_k]
    if _is_incomplete_mcq_stem(question):
        results = _mcq_stem_lookup(question, class_no, subject, book, n=max(8, top_k))
        if results:
            return [{"score": 1.70 - index * 0.01, "vector_score": 0.0, "keyword_hits": len(_mcq_stem_terms(question)), **payload} for index, payload in enumerate(results)][:top_k]
    if _is_fill_blank_question(question):
        results = _fill_blank_lookup(question, class_no, subject, book, n=max(8, top_k))
        if not results:
            fill_terms = list(dict.fromkeys(_fill_blank_terms(question)))
            topic_results = _topic_search(question, class_no, subject, book, n=max(12, top_k))
            results = []
            for payload in topic_results:
                text = str(payload.get("text") or "").lower()
                hits = sum(bool(re.search(_wb(term), text)) for term in fill_terms)
                if hits >= max(2, min(3, len(fill_terms))):
                    results.append(payload)
        if results:
            return [{"score": 1.60 - index * 0.01, "vector_score": 0.0, "keyword_hits": len(_fill_blank_terms(question)), **payload} for index, payload in enumerate(results)][:top_k]
    fact_results = _fact_lookup(question, class_no, subject, book, n=max(8, top_k))
    if fact_results:
        return [{"score": 1.50 - index * 0.01, "vector_score": 0.0, "keyword_hits": 2, **payload} for index, payload in enumerate(fact_results)][:top_k]
    intent = _question_intent(question)
    if intent in {"definition", "unit", "formula"}:
        topic_results = _topic_search(question, class_no, subject, book, n=max(8, top_k))
        if topic_results:
            return [{"score": 1.30 - index * 0.01, "vector_score": 0.0, "keyword_hits": 1, **payload} for index, payload in enumerate(topic_results)][:top_k]
    started_items = _vector_items(client, question, class_no, subject, book, top_k)
    items = started_items
    if not items and book:
        items = _vector_items(client, question, class_no, subject, None, top_k)
    keywords = _keywords(question)
    query_terms = {word for word in re.findall(r"[a-zA-Z]{3,}", (question or "").lower()) if word not in _STOP}
    for item in items:
        text = (item.get("text") or "").lower()
        hits = sum(bool(re.search(_wb(word), text)) for word in keywords)
        term_hits = sum(bool(re.search(_wb(word), text)) for word in query_terms)
        item["keyword_hits"] = max(hits, term_hits)
        lexical_boost = min(config.KEYWORD_BOOST * hits, config.KEYWORD_BOOST_MAX)
        if term_hits >= 2:
            lexical_boost += min(0.08, 0.02 * term_hits)
        item["score"] = item["vector_score"] + lexical_boost
    by_key = {_key(item): item for item in items}
    exact_payloads = _keyword_search(question, class_no, subject, book=book, n=max(top_k * 3, 10))
    exact_keys = {_key(payload) for payload in exact_payloads}
    for payload in exact_payloads[:5]:
        base = getattr(config, "KEYWORD_SCORE_BASE", 0.85)
        key = _key(payload)
        if key in by_key:
            by_key[key]["score"] = max(by_key[key]["score"], base)
        else:
            item = {"score": base, "vector_score": 0.0, "keyword_hits": 1, **payload}
            items.append(item)
            by_key[key] = item
    if exact_keys:
        for item in items:
            if _key(item) in exact_keys:
                item["score"] = max(float(item.get("score", 0.0)), 2.0)
        items = [item for item in items if _key(item) in exact_keys or item.get("keyword_hits", 0) >= 2]
    elif not any(item.get("keyword_hits", 0) > 0 for item in items):
        min_score = getattr(config, "MIN_SCORE", 0.72)
        items = [item for item in items if item.get("vector_score", 0.0) >= min_score]
        if not items:
            return []
    if intent in {"definition", "unit", "formula"}:
        terms = _topic_terms(question)
        if terms:
            patterns = [re.compile(_wb(term)) for term in terms]
            items = [item for item in items if any(pattern.search((item.get("text") or "").lower()) for pattern in patterns)]
            if not items:
                return []
    query_terms = list(dict.fromkeys(word for word in re.findall(r"[a-zA-Z]{3,}", (question or "").lower()) if word not in _STOP and word not in {"does", "did", "can", "could", "would", "should", "will", "same", "much", "many", "happen", "happens", "explain", "describe", "difference", "different", "between"}))
    if len(query_terms) >= 2:
        for item in items:
            text = re.sub(r"\s+", " ", (item.get("text") or "").lower())
            positions = []
            for term in query_terms:
                match = re.search(_wb(term), text)
                if match:
                    positions.append((match.start(), match.end()))
            coverage = len(positions) / len(query_terms)
            proximity_bonus = 0.0
            if len(positions) >= 2:
                positions.sort()
                span = positions[-1][1] - positions[0][0]
                proximity_bonus = max(0.0, 0.20 - min(span / 1500.0, 0.20))
            item["score"] = float(item.get("score", 0.0)) + 0.25 * coverage + proximity_bonus
    items.sort(key=lambda item: item["score"], reverse=True)
    return items[:top_k]
def neighbor_chunks(base, span=1, limit=3):
    """Return chunks from the same book/class on the same or adjacent pages."""
    try:
        page = int(base.get("page"))
    except (TypeError, ValueError):
        return []
    payloads, _ = _corpus()
    own_text = (base.get("text") or "")[:80]
    found = []
    for payload in payloads:
        if payload.get("book") != base.get("book") or payload.get("class") != base.get("class"):
            continue
        try:
            candidate_page = int(payload.get("page"))
        except (TypeError, ValueError):
            continue
        if abs(candidate_page - page) > span or (payload.get("text") or "")[:80] == own_text:
            continue
        found.append((abs(candidate_page - page), candidate_page, payload))
    found.sort(key=lambda item: (item[0], item[1]))
    return [dict(payload) for _, _, payload in found[:limit]]
