"""
retrieval.py
------------
Handles querying the FAISS index with a user question and returning
the most relevant chunks for prompt assembly.

SOLID:
  - S: Only responsible for retrieval — not embedding storage, not prompting
  - O: Swap retrieval strategy by subclassing BaseRetriever
  - L: Any Retriever subclass works wherever BaseRetriever is expected
  - I: Single focused method: retrieve(query) -> list[dict]
  - D: main.py depends on BaseRetriever, not FAISSRetriever directly

Flow:
    user question (str)
        → embed with MPNetEmbedder
        → search FAISSStore
        → deduplicate + rank
        → return top-K chunk dicts
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from config import TOP_K, LOG_FORMAT, LOG_LEVEL
from embedder import BaseEmbedder, MPNetEmbedder
from store import BaseVectorStore, FAISSStore

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract retriever — all retrievers must implement retrieve()."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Given a natural language query, return the top_k most
        relevant chunk dicts, each with an added 'score' field.
        """
        ...


class FAISSRetriever(BaseRetriever):
    """
    Retriever that embeds the query with MPNet and searches the FAISS index.

    Loads the index lazily on first retrieve() call so startup is fast
    and the model isn't loaded until actually needed.

    Also always injects the profile summary chunk regardless of score,
    since identity/overview questions should always have that context.
    """

    # Topics that should always be included regardless of similarity score.
    # These are small, high-signal chunks that ground every answer.
    ALWAYS_INCLUDE_TOPICS = {"summary"}

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        store: BaseVectorStore | None = None,
    ):
        # D: accepts any BaseEmbedder and BaseVectorStore
        self._embedder = embedder or MPNetEmbedder()
        self._store    = store    or FAISSStore()
        self._loaded   = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the FAISS index on first use."""
        if not self._loaded:
            log.info("Loading FAISS index for retrieval...")
            self._store.load()
            self._loaded = True

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embed the query, search FAISS, deduplicate, and return results.

        Always includes the profile summary chunk so the LLM always
        knows who Dhruv is regardless of what was asked.
        """
        self._ensure_loaded()

        log.info(f"Retrieving top {top_k} chunks for query: '{query[:80]}...'")

        # 1. Embed the query with the same model used during ingestion
        query_vector = self._embed_query(query)

        # 2. Search — fetch extra candidates to account for deduplication
        candidates = self._store.search(query_vector, top_k=top_k + 5)

        # 3. Always inject summary chunk for identity grounding
        always_included = self._get_always_included_chunks()

        # 4. Merge: always-included first, then ranked candidates (no duplicates)
        results = self._merge(always_included, candidates, top_k)

        log.info(f"Returning {len(results)} chunks")
        for r in results:
            log.debug(
                f"  [{r['score']:.3f}] {r['source']} | "
                f"{r['topic']} | {r.get('section', '')}"
            )

        return results

    # ── Private helpers ────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into a normalized vector."""
        vector = self._embedder.encode([query])
        return vector[0]  # shape: (dim,)

    def _get_always_included_chunks(self) -> list[dict]:
        """
        Pull chunks that should always be in context from the store metadata.
        Currently: the profile summary chunk.
        """
        if not hasattr(self._store, "_metadata"):
            return []

        always = []
        for chunk in self._store._metadata:
            if chunk.get("topic") in self.ALWAYS_INCLUDE_TOPICS:
                chunk_copy = dict(chunk)
                chunk_copy["score"] = 1.0   # max score — always relevant
                chunk_copy["always_included"] = True
                always.append(chunk_copy)

        return always

    def _merge(
        self,
        always: list[dict],
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Merge always-included chunks with ranked candidates.
        Deduplicates by text content so the same chunk isn't injected twice.
        """
        seen_texts: set[str] = set()
        results: list[dict] = []

        # Always-included chunks go first
        for chunk in always:
            key = chunk["text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                results.append(chunk)

        # Then ranked candidates up to top_k total
        for chunk in candidates:
            if len(results) >= top_k:
                break
            key = chunk["text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                results.append(chunk)

        return results


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    retriever = FAISSRetriever()

    test_queries = [
        "What is Dhruv's experience with RAG pipelines?",
        "What programming languages does Dhruv know?",
        "Tell me about Dhruv's work at KPMG",
        "What is Dhruv looking for in his next role?",
        "What does Dhruv think about fine-tuning vs RAG?",
    ]

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        print(f"{'─'*60}")
        chunks = retriever.retrieve(query, top_k=3)
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[{i}] score={chunk['score']:.3f} | "
                  f"topic={chunk['topic']} | source={chunk['source']}")
            print(chunk["text"][:200] + "...")
