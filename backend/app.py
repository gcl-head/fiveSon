"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as rest_router
from backend.api.ws import router as ws_router
from backend.core.settings import load_app_config
from training.orchestrator import Orchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/default.yaml"
STATIC_DIR = PROJECT_ROOT / "frontend/static"
INDEX_PATH = PROJECT_ROOT / "frontend/templates/index.html"

app_cfg = load_app_config(CONFIG_PATH)
orch = Orchestrator(CONFIG_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    worker = Thread(target=orch.run_blocking, name="orchestrator-loop", daemon=True)
    worker.start()
    try:
        yield
    finally:
        orch.request_stop()
        await asyncio.to_thread(worker.join, 5.0)


app = FastAPI(title=app_cfg.project_name, lifespan=lifespan)

app.include_router(rest_router, prefix="/api")
app.include_router(ws_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(INDEX_PATH))
