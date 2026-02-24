"""
dependencies.py
---------------
Manages singleton instances of the retriever, prompt builder,
and OpenAI client. These are created once at startup and reused
across all requests.

FastAPI's dependency injection system pulls these into route
handlers via Depends() — route handlers never instantiate
these directly.
"""

import logging
import os

from openai import OpenAI

from retrieval import FAISSRetriever
from prompt import TwinPromptBuilder

log = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
# Initialised to None — populated during app lifespan startup

_retriever: FAISSRetriever | None = None
_prompt_builder: TwinPromptBuilder | None = None
_openai_client: OpenAI | None = None


# ── Initialisation (called once from lifespan) ────────────────────────────────

def init_dependencies() -> None:
    """
    Instantiate all singletons and eagerly load the FAISS index.
    Called once during FastAPI lifespan startup before any request
    is accepted.
    """
    global _retriever, _prompt_builder, _openai_client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Copy .env.example → .env and add your key."
        )

    log.info("Initialising dependencies...")

    _openai_client  = OpenAI(api_key=api_key)
    _retriever      = FAISSRetriever()
    _prompt_builder = TwinPromptBuilder()

    # Eagerly load FAISS index so the first request is not slow
    log.info("Pre-loading FAISS index...")
    _retriever._ensure_loaded()

    log.info("All dependencies ready.")


# ── FastAPI dependency functions ──────────────────────────────────────────────
# These are passed to Depends() in route handlers.
# Each raises clearly if called before init_dependencies().

def get_retriever() -> FAISSRetriever:
    if _retriever is None:
        raise RuntimeError("Retriever not initialised. Was init_dependencies() called?")
    return _retriever


def get_prompt_builder() -> TwinPromptBuilder:
    if _prompt_builder is None:
        raise RuntimeError("PromptBuilder not initialised. Was init_dependencies() called?")
    return _prompt_builder


def get_openai_client() -> OpenAI:
    if _openai_client is None:
        raise RuntimeError("OpenAI client not initialised. Was init_dependencies() called?")
    return _openai_client
