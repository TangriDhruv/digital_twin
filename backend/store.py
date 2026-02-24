"""
store.py
--------
Handles building, saving, and loading the FAISS vector index
alongside its associated chunk metadata.


"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np

from config import INDEX_PATH, METADATA_PATH, TOP_K

log = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def build(self, embeddings: np.ndarray) -> None:
        """Build index from embeddings matrix."""
        ...

    @abstractmethod
    def save(self) -> None:
        """Persist index and metadata to disk."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load index and metadata from disk."""
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = TOP_K) -> list[dict]:
        """
        Search index for nearest neighbours.
        Returns list of chunk dicts with an added 'score' field.
        """
        ...


class FAISSStore(BaseVectorStore):
    """
    FAISS-backed vector store using IndexFlatIP (exact inner product).
    Cosine similarity is achieved by using normalized embeddings.

    Metadata (chunk dicts) is stored separately as a pickle file,
    indexed in the same order as the FAISS vectors.
    """

    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        self._index_path    = index_path
        self._metadata_path = metadata_path
        self._index: faiss.Index | None = None
        self._metadata: list[dict] = []

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Build a fresh FAISS index from an embeddings matrix.
        Metadata list must be the same length and order as embeddings rows.
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
                "must be the same length."
            )

        dim = embeddings.shape[1]
        log.info(f"Building FAISS IndexFlatIP (dim={dim}, vectors={len(embeddings)})")

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._metadata = metadata

        log.info(f"Index built — {self._index.ntotal} vectors stored")

    # ── Persist ────────────────────────────────────────────────────────────

    def save(self) -> None:
        if self._index is None:
            raise RuntimeError("No index to save. Call build() first.")

        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self._index_path))
        log.info(f"FAISS index saved → {self._index_path}")

        with open(self._metadata_path, "wb") as f:
            pickle.dump(self._metadata, f)
        log.info(f"Metadata saved    → {self._metadata_path}")

    # ── Load ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self._index_path}. Run ingest.py first."
            )
        if not self._metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {self._metadata_path}. Run ingest.py first."
            )

        self._index = faiss.read_index(str(self._index_path))
        log.info(f"FAISS index loaded ← {self._index_path} ({self._index.ntotal} vectors)")

        with open(self._metadata_path, "rb") as f:
            self._metadata = pickle.load(f)
        log.info(f"Metadata loaded   ← {self._metadata_path} ({len(self._metadata)} chunks)")

    # ── Search ─────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = TOP_K) -> list[dict]:
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        # FAISS expects shape (1, dim)
        query = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self._index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for empty slots
            chunk = dict(self._metadata[idx])   # copy to avoid mutation
            chunk["score"] = float(score)
            results.append(chunk)

        return results
