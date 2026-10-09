from src.rag.retriever import _corpus

payloads, lower = _corpus()

for p, t in zip(payloads, lower):
    if "nichrome" in t:
        print("BOOK:", p.get("book"), "| PAGE:", p.get("page"))
        print(t[:1200])
        print("-" * 70)