import sys
from collections import Counter
import config
from src.rag.retriever import _client, retrieve
from src.rag.rag_pipeline import _rewrite_query
if __name__ == "__main__":
    c = _client()
    pts, off, books = [], None, Counter()
    while True:
        p, off = c.scroll(config.COLLECTION, limit=1000, offset=off, with_payload=["book"], with_vectors=False)
        books.update(x.payload.get("book") for x in p)
        if off is None:
            break
    print("=== Indexed books (chunks) ===")
    for b, n in sorted(books.items(), key=lambda x: str(x[0])):
        print(f"{n:5d}  {b}")
    q = " ".join(sys.argv[1:]) or "Why does land heat up faster than water?"
    eq = _rewrite_query(q)
    print(f"\nQuery: {q}\nRewritten: {eq}\nMIN_SCORE = {config.MIN_SCORE}\n")
    try:
        results = retrieve(eq, book="Curiosity")
    except TypeError:
        results = retrieve(eq)
    print("\n=== Retrieval Results ===")
    if not results:
        print("NO RESULTS")
    for r in results:
        ok = "PASS" if r["score"] >= config.MIN_SCORE else "fail"
        print(f"{ok}  score={r['score']:.3f} (vec={r.get('vector_score', 0.0):.3f})  {r['book']} p.{r['page']}")
        print("      ", (r.get("text") or "")[:300].replace("\n", " "))