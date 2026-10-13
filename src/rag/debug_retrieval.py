import sys
from collections import Counter
import config
from src.rag.retriever import _client
from src.rag.rag_pipeline import (
    _rewrite_query,
    _search,
    _prepare_chunks,
    _filter_relevant_chunks,
    answer_stream,
)

# Usage (from project root):
#   python -m src.rag.debug_retrieval "Which locker numbers remain open when 100 people toggle the lockers?"

if __name__ == "__main__":
    c = _client()
    off, books = None, Counter()
    while True:
        p, off = c.scroll(config.COLLECTION, limit=1000, offset=off, with_payload=["book"], with_vectors=False)
        books.update(x.payload.get("book") for x in p)
        if off is None:
            break
    print("=== Indexed books (chunks) ===")
    for b, n in sorted(books.items(), key=lambda x: str(x[0])):
        print(f"{n:5d}  {b}")

    q = " ".join(sys.argv[1:]) or "Which locker numbers remain open when 100 people toggle the lockers according to the rules in the chapter?"
    print(f"\nQuery: {q}")
    print(f"Rewritten: {_rewrite_query(q)}")
    print(f"MIN_SCORE = {config.MIN_SCORE}")
    print(f"LLM_MODEL = {getattr(config, 'LLM_MODEL', getattr(config, 'OLLAMA_MODEL', '?'))}")
    print(f"TEMPERATURE = {getattr(config, 'LLM_TEMPERATURE', '?')}  REPEAT_PENALTY = {getattr(config, 'REPEAT_PENALTY', '?')}  MAX_TOKENS = {getattr(config, 'MAX_TOKENS', '?')}")

    # Same steps the app uses (no hard-coded book filter)
    chunks = _search(q)
    chunks = _prepare_chunks(q, chunks)
    chunks = _filter_relevant_chunks(q, chunks)

    print(f"\n=== Chunks sent to the LLM: {len(chunks)} ===")
    for i, r in enumerate(chunks, 1):
        print(f"\n--- [{i}] {r.get('book')} p.{r.get('page')}  score={float(r.get('score', 0.0) or 0.0):.3f}")
        print(r.get("text") or "")   # FULL text, so broken words from the PDF are visible

    print("\n=== Final answer from answer_stream ===")
    for kind, val in answer_stream(q):
        if kind == "token":
            print(val, end="", flush=True)
        elif kind == "sources":
            print(f"[sources: {val}]")
    print()