"""Central configuration for the CBSE RAG app — latency-tuned defaults."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BOOKS_DIR = BASE_DIR / "data" / "books"
PAPERS_DIR = BASE_DIR / "data" / "papers"
QDRANT_PATH = str(BASE_DIR / "data" / "qdrant_db")
COLLECTION = "cbse_books"

# Multilingual English / Hindi / Marathi / Sanskrit embeddings.
EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_DIM = 384

# Chunking: keep unchanged so existing indexed data remains compatible.
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
MIN_PAGE_CHARS = 40
OCR_LANGS = "eng+hin+mar+san"

# Retrieval: preserve candidate recall while skipping optional query rewriting.
TOP_K = 4
CANDIDATES = 8
MIN_SCORE = 0.72
REL_MARGIN = 0.15
KEYWORD_BOOST = 0.02
KEYWORD_BOOST_MAX = 0.10
KEYWORD_SCORE_BASE = 0.85
REWRITE_QUERY = False

# Use the model already installed locally.
OLLAMA_URL = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5:7b"
LLM_TEMPERATURE = 0.0
LLM_NUM_CTX = 3072
REPEAT_PENALTY = 1.02
MAX_TOKENS = 120
LLM_TIMEOUT = 120

HOST = "127.0.0.1"
PORT = 5000
PAPER_CONTEXT_CHUNKS = 8
PAPER_TEMPERATURE = 0.5
