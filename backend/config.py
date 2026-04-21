"""
backend/config.py
─────────────────
Single source of truth for every configurable value.
All secrets come from the .env file — nothing is hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# ─── Database ─────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME",     "rag"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
}

# ─── JWT ──────────────────────────────────────────────────────────────────────
JWT_SECRET_KEY                  = os.getenv("JWT_SECRET_KEY", "change-me")
JWT_ALGORITHM                   = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ─── Paths ────────────────────────────────────────────────────────────────────
JSON_DATA_DIR       = os.getenv("JSON_DATA_DIR",       "ctg-studies.json")
FAISS_INDEX_PATH    = os.getenv("FAISS_INDEX_PATH",    "backend/embeddings_output/trial_faiss.index")
EMBEDDINGS_PKL_PATH = os.getenv("EMBEDDINGS_PKL_PATH", "backend/embeddings_output/df_with_embeddings.pkl")
BATCH_SIZE          = int(os.getenv("BATCH_SIZE", "1000"))

# ─── OpenAI ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4o-mini"

# ─── RAG knobs ────────────────────────────────────────────────────────────────
# L2 distance threshold — results with distance > this are discarded.
# Lower = stricter. Tune empirically against your embedding space.
FAISS_SIMILARITY_THRESHOLD = float(os.getenv("FAISS_SIMILARITY_THRESHOLD", "60.0"))
RETRIEVAL_K      = int(os.getenv("RETRIEVAL_K",      "5"))
SEARCH_MULTIPLIER = int(os.getenv("SEARCH_MULTIPLIER", "10"))

# ─── App ──────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")