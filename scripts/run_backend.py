"""Run FastAPI backend server."""

from __future__ import annotations

from pathlib import Path

import uvicorn

from backend.core.settings import load_app_config


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_app_config(project_root / "configs/default.yaml")
    uvicorn.run("backend.app:app", host=cfg.host, port=cfg.port, reload=False)


if __name__ == "__main__":
    main()
