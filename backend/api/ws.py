"""WebSocket endpoint for live runtime status updates."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.runtime import runtime_registry

router = APIRouter()


@router.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(runtime_registry.snapshot())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
