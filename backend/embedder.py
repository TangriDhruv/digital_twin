"""
embedder.py
-----------
Handles loading the embedding model and encoding text into vectors.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, NORMALIZE_EMBEDDINGS

log = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract embedder — all embedders must implement encode()."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension. Required for FAISS index creation."""
        ...

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings into a float32 numpy array.
        Shape: (len(texts), self.dimension)
        """
        ...


class MPNetEmbedder(BaseEmbedder):
    """
    Embedder using sentence-transformers/all-mpnet-base-v2.
    Embeddings are L2-normalized so inner product == cosine similarity.
    Lazy-loads the model on first call to encode().
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        normalize: bool = NORMALIZE_EMBEDDINGS,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize  = normalize
        self._model: SentenceTransformer | None = None
        self._dim: int | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            log.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            # Probe dimension with a dummy encode
            probe = self._model.encode(["probe"], normalize_embeddings=self._normalize)
            self._dim = probe.shape[1]
            log.info(f"Model loaded — embedding dim: {self._dim}")
        return self._model

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        log.info(f"Encoding {len(texts)} chunks (batch_size={self._batch_size})")
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=self._normalize,
        )
        return np.array(embeddings).astype("float32")
