"""
ingest.py
---------
Ingestion orchestrator. Wires together loaders, embedder, and vector
store to build the FAISS index from personal data + GitHub activity.

Pipeline:
  loaders → chunks → PII redaction → embed → FAISS store

Usage:
    python backend/ingest.py
"""

import logging
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from config import DOCS_DIR, PROFILE_PATH, LOG_FORMAT, LOG_LEVEL
from loaders import BaseLoader, JSONProfileLoader, MarkdownLoader
from github_loader import GitHubLoader
from embedder import OpenAIEmbedder
from store import FAISSStore
from pii_redactor import PIIRedactor

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)


def collect_chunks(loaders: list[tuple[BaseLoader, Path | None]]) -> list[dict]:
    all_chunks: list[dict] = []
    for loader, path in loaders:
        chunks = loader.load(path)
        all_chunks.extend(chunks)
    return all_chunks


def print_summary(chunks: list[dict]) -> None:
    counts = Counter(c["topic"] for c in chunks)
    print("\n── Ingestion Summary ─────────────────────────────")
    for topic, count in sorted(counts.items()):
        print(f"  {topic:<40} {count} chunks")
    print(f"  {'TOTAL':<40} {len(chunks)} chunks")

    redacted = [c for c in chunks if c.get("pii_redacted")]
    if redacted:
        print(f"\n  ── PII Redacted in {len(redacted)} chunks ──────────")
        for c in redacted:
            print(f"  [{c.get('topic', '?')} / {c.get('section', '?')}] → {c['pii_redacted']}")
    print("──────────────────────────────────────────────────\n")


def run_ingestion() -> None:
    log.info("Starting ingestion pipeline...")

    loaders: list[tuple[BaseLoader, Path | None]] = [
        (JSONProfileLoader(), PROFILE_PATH),
        *[(MarkdownLoader(), md) for md in sorted(DOCS_DIR.glob("*.md"))],
        (GitHubLoader(), None),
    ]

    # 1. Load all chunks
    all_chunks = collect_chunks(loaders)
    log.info(f"Total chunks collected: {len(all_chunks)}")

    # 2. Redact PII before embedding — nothing sensitive goes into the index
    redactor   = PIIRedactor()
    all_chunks = redactor.redact_chunks(all_chunks)

    # 3. Embed and store
    embedder   = OpenAIEmbedder()
    embeddings = embedder.encode([c["text"] for c in all_chunks])

    store = FAISSStore()
    store.build(embeddings, all_chunks)
    store.save()

    print_summary(all_chunks)
    log.info("Ingestion complete")


if __name__ == "__main__":
    run_ingestion()