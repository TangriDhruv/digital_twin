"""
retrieval.py
------------
Retrieves relevant chunks from FAISS for a given query.
Embeds queries using OpenAI text-embedding-3-small.
"""

import logging
from abc import ABC, abstractmethod

from embedder import OpenAIEmbedder
from store import FAISSStore
from config import TOP_K

log = logging.getLogger(__name__)


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        pass


class FAISSRetriever(BaseRetriever):

    def __init__(self):
        self._embedder = OpenAIEmbedder()
        self._store    = FAISSStore()
        self._loaded   = False

    def _ensure_loaded(self):
        if not self._loaded:
            log.info("Loading FAISS index for retrieval.")
            self._store.load()
            self._loaded = True

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        self._ensure_loaded()

        query_vector = self._embedder.encode([query])[0]
        candidates   = self._store.search(query_vector, top_k=top_k + 5)

        # Always inject profile summary for identity grounding
        summary_chunks = [
            c for c in self._store._metadata
            if c.get("topic") == "summary"
        ]

        # Deduplicate by text
        seen   = set()
        chunks = []

        for c in summary_chunks:
            key = c["text"][:100]
            if key not in seen:
                seen.add(key)
                chunks.append(c)

        for c in candidates:
            key = c["text"][:100]
            if key not in seen:
                seen.add(key)
                chunks.append(c)
                if len(chunks) >= top_k + len(summary_chunks):
                    break

        log.info(f"Retrieving top {top_k} chunks for query: '{query[:50]}'")
        return chunks[:top_k + len(summary_chunks)]