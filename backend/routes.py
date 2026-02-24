"""
routes.py
---------
FastAPI route handlers. Each endpoint has exactly one job.
No business logic lives here — routes only coordinate between
dependencies and return responses.

SOLID:
  - S: Each route handles one HTTP endpoint
  - D: Routes depend on injected abstractions via Depends(),
       never on concrete classes directly
"""

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from openai import OpenAI

from config import TOP_K
from schemas import ChatRequest, HealthResponse, ConfigResponse
from dependencies import get_retriever, get_prompt_builder, get_openai_client
from retrieval import FAISSRetriever
from prompt import TwinPromptBuilder

log = logging.getLogger(__name__)

router = APIRouter()


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    """Liveness check — confirms the server is running."""
    return {"status": "ok"}


# ── GET /config ───────────────────────────────────────────────────────────────

@router.get("/config", response_model=ConfigResponse)
def get_config(
    retriever: FAISSRetriever = Depends(get_retriever),
):
    """Returns non-secret runtime config. Useful for debugging."""
    return {
        "model":        "gpt-4o-mini",
        "top_k":        TOP_K,
        "index_loaded": retriever._loaded,
    }


# ── POST /chat ────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(
    request: ChatRequest,
    retriever:      FAISSRetriever   = Depends(get_retriever),
    prompt_builder: TwinPromptBuilder = Depends(get_prompt_builder),
    openai_client:  OpenAI           = Depends(get_openai_client),
):
    """
    Main chat endpoint. Streams the response as Server-Sent Events (SSE).

    Flow:
        1. Retrieve relevant chunks from FAISS
        2. Assemble prompt with context + history
        3. Stream OpenAI response token by token back to the client

    SSE format:
        data: {"token": "..."}  — one per token
        data: {"done": true}    — signals end of stream
        data: {"error": "..."}  — if something goes wrong mid-stream
    """
    return StreamingResponse(
        _stream(request, retriever, prompt_builder, openai_client),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",    # prevents nginx from buffering the stream
        },
    )


# ── Streaming generator ───────────────────────────────────────────────────────

async def _stream(
    request:        ChatRequest,
    retriever:      FAISSRetriever,
    prompt_builder: TwinPromptBuilder,
    openai_client:  OpenAI,
):
    """
    Private async generator that drives the streaming pipeline.
    Separated from the route handler so the route stays clean
    and this logic is independently testable.
    """
    try:
        # 1. Retrieve relevant chunks
        log.info(f"Chat: '{request.message[:60]}'")
        chunks = retriever.retrieve(request.message, top_k=request.top_k)

        # 2. Build prompt
        history_dicts = [m.model_dump() for m in request.history]
        messages = prompt_builder.build(
            query=request.message,
            chunks=chunks,
            history=history_dicts,
        )

        # 3. Stream from OpenAI token by token
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1024,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'token': delta.content})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

    except FileNotFoundError:
        log.error("FAISS index not found")
        yield f"data: {json.dumps({'error': 'Index not found. Run python backend/ingest.py first.'})}\n\n"

    except Exception as e:
        log.error(f"Stream error: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': 'Something went wrong. Check server logs.'})}\n\n"
