import re


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int, overlap: int):
    """Paragraph-aware sliding window chunker."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:  # PDFs often use single newlines only
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur}\n{p}".strip()
            continue
        if cur:
            chunks.append(cur)
        while len(p) > size:
            chunks.append(p[:size])
            p = p[size - overlap:]
        cur = (cur[-overlap:] + "\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 30]


def parse_class(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def parse_subject(folder: str, class_no=None) -> str:
    s = re.sub(r"^\s*\d+(st|nd|rd|th)?\s*", "", folder, flags=re.I).strip()
    return s or folder
