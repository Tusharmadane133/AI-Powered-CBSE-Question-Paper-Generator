"""Central configuration for the CBSE RAG app."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BOOKS_DIR = BASE_DIR / "data" / "books"
PAPERS_DIR = BASE_DIR / "data" / "papers"
QDRANT_PATH = str(BASE_DIR / "data" / "qdrant_db")
COLLECTION = "cbse_books"

# --- Embeddings (multilingual: English / Hindi / Marathi / Sanskrit) ---
EMBED_MODEL = "intfloat/multilingual-e5-small"   # upgrade: "intfloat/multilingual-e5-base" or "BAAI/bge-m3"
EMBED_DIM = 384                                  # e5-small=384, e5-base=768, bge-m3=1024

# --- Chunking ---
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
MIN_PAGE_CHARS = 40          # below this a page is treated as scanned -> OCR
OCR_LANGS = "eng+hin+mar+san"

# --- Retrieval ---
TOP_K = 10                   # chunks sent to the LLM (was 8)
CANDIDATES = 24              # chunks fetched from Qdrant before keyword re-ranking
MIN_SCORE = 0.72             # was 0.78 (too strict -> false "not found"); tune with debug_retrieval.py
REL_MARGIN = 0.15            # keep only chunks within this score distance of the best chunk (was 0.10)
KEYWORD_BOOST = 0.02         # added to score per query keyword found in the chunk text
KEYWORD_BOOST_MAX = 0.10
KEYWORD_SCORE_BASE = 0.85    # score given to chunks found by exact rare-word match (e.g. "albedo")
REWRITE_QUERY = True         # LLM rewrites Marathi/Hindi/Hinglish question into English search keywords

# --- LLM via Ollama ---
OLLAMA_URL = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5:7b"     # alternatives: llama3.1:8b, gemma2:9b, qwen2.5:3b (low RAM)
LLM_TEMPERATURE = 0.0
LLM_NUM_CTX = 8192
REPEAT_PENALTY = 1.15        # fixes repetition loops (e.g. "aavesh var aatmakta ...")
MAX_TOKENS = 800             # max tokens per chat answer
LLM_TIMEOUT = 600

# --- Web ---
HOST = "127.0.0.1"
PORT = 5000

# --- Question paper generator ---
PAPER_CONTEXT_CHUNKS = 8     # textbook chunks given to the LLM per generation call
PAPER_TEMPERATURE = 0.5