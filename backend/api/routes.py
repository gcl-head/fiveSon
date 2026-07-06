"""REST endpoints for runtime status and control."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.runtime import runtime_registry
from backend.services.game_service import game_service

router = APIRouter()


class MoveRequest(BaseModel):
    """Request body for player move on the board."""

    move: int


class QuickEvalRequest(BaseModel):
    """Request body for quick validation matches."""

    games: int = 30
    generation: int | None = None
    baseline_generation: int = 0


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status")
def status() -> dict[str, object]:
    return runtime_registry.snapshot()


@router.post("/control/pause")
def pause_training() -> dict[str, str]:
    runtime_registry.update(status="paused", paused=True)
    return {"message": "training paused"}


@router.post("/control/resume")
def resume_training() -> dict[str, str]:
    runtime_registry.update(
        status="idle",
        paused=False,
        auto_paused=False,
        overload_streak=0,
        last_overload_reason="",
    )
    return {"message": "training resumed"}


@router.post("/control/parallel/{count}")
def set_parallel_games(count: int) -> dict[str, object]:
    allowed = {1, 4, 8, 16, 32, 64}
    if count not in allowed:
        raise HTTPException(status_code=400, detail=f"parallel count must be one of {sorted(allowed)}")

    runtime_registry.update(target_parallel_self_play_games=count)
    return {
        "message": "parallel self-play target updated",
        "target_parallel_self_play_games": count,
    }


@router.get("/game/state")
def game_state() -> dict[str, object]:
    return game_service.state()


@router.post("/game/reset")
def game_reset() -> dict[str, object]:
    return game_service.reset()


@router.post("/game/move")
def game_move(req: MoveRequest) -> dict[str, object]:
    try:
        result = game_service.play_human_move(req.move)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    game_meta = game_service.state()
    return {
        "board": result.board,
        "to_play": result.to_play,
        "winner": result.winner,
        "ai_move": result.ai_move,
        "legal_moves": result.legal_moves,
        "message": result.message,
        "deployed_generation": game_meta.get("deployed_generation", 0),
        "ai_policy": game_meta.get("ai_policy", "heuristic_generation"),
    }


@router.post("/eval/quick")
def quick_eval(req: QuickEvalRequest) -> dict[str, object]:
    try:
        result = game_service.quick_eval(
            games=req.games,
            generation=req.generation,
            baseline_generation=req.baseline_generation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "games": result.games,
        "generation": result.generation,
        "baseline_generation": result.baseline_generation,
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "win_rate": result.win_rate,
        "avg_moves": result.avg_moves,
        "avg_game_ms": result.avg_game_ms,
    }
