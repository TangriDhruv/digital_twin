"""
main.py
-------
App entry point. Creates the FastAPI app, registers middleware,
wires the lifespan, and mounts routes.
All logic lives in routes.py, dependencies.py, schemas.py.

Usage:
    uvicorn backend.main:app --reload --port 8000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import LOG_FORMAT, LOG_LEVEL
from dependencies import init_dependencies
from routes import router

load_dotenv()
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise all dependencies. Shutdown: nothing to clean up."""
    init_dependencies()
    yield
    log.info("Server shutting down.")


app = FastAPI(
    title="Dhruv Tangri — Digital Twin API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://localhost:3000",   # CRA default
        "https://digital-twin-wine.vercel.app/",
    ],
    allow_origin_regex="https://.*\\.vercel\\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
