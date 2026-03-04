"""
embedder.py
-----------
Embeds text using OpenAI's text-embedding-3-small model.
"""

import logging
import os
import numpy as np
from abc import ABC, abstractmethod
from openai import OpenAI
from config import EMBEDDING_BATCH_SIZE

log = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        pass


class OpenAIEmbedder(BaseEmbedder):
    """
    Uses OpenAI text-embedding-3-small.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self._model  = model
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def encode(self, texts: list[str]) -> np.ndarray:
        log.info(f"Embedding {len(texts)} texts via OpenAI")

        
        texts = [t.replace("\n", " ") for t in texts]

        all_embeddings = []
        batch_size = EMBEDDING_BATCH_SIZE

        for i in range(0, len(texts), batch_size):
            batch    = texts[i:i + batch_size]
            response = self._client.embeddings.create(
                input=batch,
                model=self._model,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings, dtype=np.float32)

        # L2 normalize so dot product = cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms