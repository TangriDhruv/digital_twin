"""
config.py
---------
Central configuration for the entire digital twin pipeline.
All paths, model names, and hyperparameters live here.
Nothing else should hardcode these values.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
DOCS_DIR    = DATA_DIR / "docs"
PROFILE_PATH = DATA_DIR / "profile.json"
INDEX_DIR   = ROOT_DIR / "faiss_index"
INDEX_PATH  = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

# ── Embedding model ───────────────────────────────────────────────────────────

EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True   # enables cosine similarity via dot product

# ── Chunking ──────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 300   # target word count per chunk
CHUNK_OVERLAP = 50    # word overlap between consecutive chunks

# ── FAISS ─────────────────────────────────────────────────────────────────────

# IndexFlatIP = exact inner product search
# cosine similarity when embeddings are normalized
FAISS_INDEX_TYPE = "IndexFlatIP"

# ── Retrieval (used later in retrieval.py) ────────────────────────────────────

TOP_K = 5   # number of chunks to retrieve per query

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(levelname)s | %(name)s | %(message)s"
LOG_LEVEL  = "INFO"

# ── GitHub Integration ────────────────────────────────────────────────────────

GITHUB_USERNAME    = "TangriDhruv"           
GITHUB_MAX_REPOS   = 10                       # max repos to fetch
GITHUB_MAX_COMMITS = 30                       # max commits per repo
GITHUB_API_BASE    = "https://api.github.com"
