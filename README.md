# AI-Powered CBSE Question Paper Generator

Generates CBSE-style question papers (with answer key, Word + PDF export) for Class 8, 9, 10 from the textbook PDFs, and also answers questions from the books. Fully local RAG: PDF -> chunks -> multilingual embeddings -> Qdrant -> Ollama LLM -> Flask web app (Python).

## 1. Setup
    python -m venv venv
    venv\Scripts\activate          (Linux/Mac: source venv/bin/activate)
    pip install -r requirements.txt

Install Ollama from https://ollama.com then:

    ollama pull qwen2.5:7b         (low RAM <8GB: ollama pull qwen2.5:3b and change LLM_MODEL in config.py)

## 2. Add books
Copy your folders into `data/books/` like this (empty subject folders already exist):

    data/books/8th Book/8th Science/*.pdf
    data/books/9th Book/9th Math/*.pdf
    data/books/10th Book/10th Mathematics/*.pdf

## 3. Index books (once, and again after adding PDFs)
    python -m src.ingestion.ingest          (add --reset to rebuild from scratch)

Scanned pages fall back to OCR if Tesseract (+ eng/hin/mar/san language data) is installed.

## 4. Run
    ollama serve                            (if not already running)
    python app.py                           -> http://127.0.0.1:5000

IMPORTANT: Qdrant local mode allows one process at a time. Stop `python app.py` before running the ingestion.

## 5. Generate a question paper
1. Open the "Generate Paper" tab, choose Class and Subject.
2. Set school name, exam title, duration, difficulty, language. Optional: chapters/topics (empty = whole book).
3. Edit the paper pattern (question type, number of questions, marks each). Default = 50 marks:
   10 MCQ x1, 5 very short x2, 5 short x3, 3 long x5. Types: MCQ, very short, short, long, case-based.
4. Click "Generate Question Paper" (takes a few minutes on CPU; progress is shown).
5. "Download Word (.docx)" = editable paper + answer key. "Print / Save as PDF" opens a print view -> choose Save as PDF.
   Papers are saved in data/papers/ and listed under "Previous papers".

## Tuning (config.py)
- MIN_SCORE: raise to reduce wrong answers in "Ask Books", lower if it says "not found" too often.
- EMBED_MODEL: multilingual-e5-base or BAAI/bge-m3 for better Marathi/Hindi (update EMBED_DIM, re-ingest with --reset).
- LLM_MODEL: qwen2.5:7b (best multilingual balance), llama3.1:8b, gemma2:9b. Low RAM: qwen2.5:3b.
- PAPER_CONTEXT_CHUNKS / LLM_NUM_CTX: more context per generation call (needs more RAM).
