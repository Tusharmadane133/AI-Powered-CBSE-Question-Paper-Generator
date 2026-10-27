"""Ingest PDFs: data/books/<class folder>/<subject folder>/*.pdf -> Qdrant.
Run: python -m src.ingestion.ingest [--reset]
"""
import argparse
import hashlib
import io
import re
import sys
import uuid
from pathlib import Path

import fitz
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

import config
from src.rag.embeddings import embed_passages
from src.utils.text_utils import chunk_text, clean_text, parse_class, parse_subject


def _text_quality(text: str) -> float:
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    allowed = set(".,;:!?()[]{}+-=/<>%°×²³")
    allowed.update({"'", '"'})
    useful = sum(1 for c in chars if c.isalnum() or c in allowed)
    return useful / len(chars)


def ocr_page(page) -> str:
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=250, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(
            img,
            lang=config.OCR_LANGS,
            config="--psm 6",
        )
    except Exception as e:
        print(f"  [ocr skipped] {e}", file=sys.stderr)
        return ""


def extract_printed_page(page, text: str):
    """Best-effort extraction of the page number printed on the textbook page.

    Keep this separate from the PDF viewer page index: `page` is the PDF page
    used to open the document, while `printed_page` is only a citation label.
    Returns None when a reliable-looking printed page number is not found.
    """
    candidates = []
    try:
        page_dict = page.get_text("dict") or {}
        height = float(page.rect.height or 0)
        width = float(page.rect.width or 0)
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    value = str(span.get("text", "")).strip()
                    if not re.fullmatch(r"\d{1,4}", value):
                        continue
                    number = int(value)
                    if number < 1 or number > 2000:
                        continue
                    bbox = span.get("bbox") or (0, 0, 0, 0)
                    y0, y1 = float(bbox[1]), float(bbox[3])
                    x0, x1 = float(bbox[0]), float(bbox[2])
                    # Page numbers normally sit close to the top or bottom edge.
                    edge_distance = min(y0, max(0.0, height - y1))
                    if height and edge_distance <= height * 0.14:
                        # Prefer numbers closer to an edge, then numbers near a corner.
                        corner_bonus = 0 if (x0 < width * 0.18 or x1 > width * 0.82) else 1
                        candidates.append((edge_distance, corner_bonus, number))
    except Exception:
        pass

    if candidates:
        candidates.sort()
        return candidates[0][2]

    # Fallback for OCR/text extraction where the page number is an isolated line.
    lines = [line.strip() for line in (text or "").splitlines()]
    isolated = [
        int(line) for line in lines
        if re.fullmatch(r"\d{1,4}", line) and 1 <= int(line) <= 2000
    ]
    if len(isolated) == 1:
        return isolated[0]
    return None


def extract_pages(pdf_path: Path):
    doc = fitz.open(pdf_path)
    try:
        for pdf_page_no, page in enumerate(doc, start=1):
            extracted = clean_text(page.get_text("text") or "")
            text = extracted

            if (
                len(extracted.strip()) < config.MIN_PAGE_CHARS
                or _text_quality(extracted) < 0.65
            ):
                ocr_text = clean_text(ocr_page(page))
                if ocr_text and (
                    not extracted
                    or _text_quality(ocr_text) > _text_quality(extracted)
                ):
                    text = ocr_text

            if text:
                printed_page = extract_printed_page(page, text)
                yield pdf_page_no, printed_page, text
    finally:
        doc.close()


def discover_pdfs():
    for pdf in sorted(config.BOOKS_DIR.rglob("*.pdf")):
        rel = pdf.relative_to(config.BOOKS_DIR).parts

        if len(rel) < 3:
            print(f"[skip] {pdf} (expected <class>/<subject>/file.pdf)")
            continue

        try:
            class_no = parse_class(rel[0])
            subject = parse_subject(rel[1])
        except Exception as e:
            print(f"[skip] {pdf} (could not parse class/subject: {e})")
            continue

        yield pdf, class_no, subject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop and rebuild the collection",
    )
    args = parser.parse_args()

    client = QdrantClient(path=config.QDRANT_PATH)
    try:
        exists = client.collection_exists(config.COLLECTION)

        if args.reset and exists:
            client.delete_collection(config.COLLECTION)
            exists = False

        if not exists:
            client.create_collection(
                config.COLLECTION,
                vectors_config=VectorParams(
                    size=config.EMBED_DIM,
                    distance=Distance.COSINE,
                ),
            )

        pdfs = list(discover_pdfs())
        if not pdfs:
            print(
                f"No PDFs found in {config.BOOKS_DIR}. "
                "Copy your books there first."
            )
            return

        total_pages = 0
        total_chunks = 0

        for pdf, class_no, subject in pdfs:
            print(f"\n>> Class {class_no} | {subject} | {pdf.name}")
            texts = []
            metas = []
            pages = 0

            for pdf_page_no, printed_page, text in extract_pages(pdf):
                pages += 1
                for chunk in chunk_text(
                    text,
                    config.CHUNK_CHARS,
                    config.CHUNK_OVERLAP,
                ):
                    chunk = clean_text(chunk)
                    if not chunk:
                        continue
                    texts.append(chunk)
                    metas.append(
                        {
                            "class": class_no,
                            "subject": subject,
                            "book": pdf.stem,
                            "page": pdf_page_no,
                            "printed_page": printed_page,
                            "text": chunk,
                        }
                    )

            print(f"   pages read: {pages}")
            print(f"   chunks prepared: {len(texts)}")

            for start in tqdm(range(0, len(texts), 64), desc="embedding"):
                batch_texts = texts[start:start + 64]
                batch_metas = metas[start:start + 64]
                vectors = embed_passages(batch_texts)
                points = []

                for vector, meta in zip(vectors, batch_metas):
                    raw_id = hashlib.md5(
                        (
                            f"{pdf.resolve()}|"
                            f"{meta['page']}|"
                            f"{meta['text']}"
                        ).encode("utf-8")
                    ).hexdigest()
                    point_id = str(uuid.UUID(raw_id))
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector.tolist(),
                            payload=meta,
                        )
                    )

                if points:
                    client.upsert(config.COLLECTION, points)

            total_pages += pages
            total_chunks += len(texts)
            print(f"   {len(texts)} chunks stored")

        print(f"\nDone. {total_pages} pages processed.")
        print(f"Done. {total_chunks} chunks indexed.")
        print(f"Collection: {config.COLLECTION}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
