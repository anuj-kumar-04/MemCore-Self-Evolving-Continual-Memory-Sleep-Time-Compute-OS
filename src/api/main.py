"""
MemCore — FastAPI Application Gateway & Lifespan Hooks
"""

from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import structlog

from src.api.routes_memory import router as memory_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup hooks
    logger.info("memcore_os_booting", version="1.0.0")
    # Verify connections or fallback initialization
    yield
    # Shutdown hooks
    logger.info("memcore_os_shutting_down")


app = FastAPI(
    title="MemCore — Self-Evolving Continual Memory & Sleep-Time Compute OS",
    description=(
        "Distributed cognitive memory architecture solving Temporal Contradiction & Fact Drift, "
        "Context Bloat, and Zero Procedural Learning via Bi-Temporal Knowledge Graphs and Sleep-Time Compute."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local dashboards and IDE webviews
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Core Memory OS Routes
app.include_router(memory_router, prefix="/api/v1")

# Mount UI static files if present
ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.isdir(ui_dir):
    app.mount("/static", StaticFiles(directory=ui_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui_root():
    """
    Serves the MemCore Neuro-Visualizer Dashboard.
    """
    index_path = os.path.join(ui_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>MemCore API Gateway Running. Navigate to /docs for OpenAPI specs.</h2>")


@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    return await serve_ui_root()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "MemCore Sleep-Time Compute OS",
        "architecture": {
            "tier_1": "Redis Sorted Sets (Working Memory Sliding Window)",
            "tier_2": "PostgreSQL 16 + pgvector HNSW (Bi-Temporal Knowledge Graph)",
            "tier_3": "PostgreSQL JSONB + Redis Cache (Procedural Playbook)",
            "fast_path": "<50ms Context Assembly via Submodular Pruning",
            "sleep_daemon": "Celery Worker + Graph Diff Engine"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
