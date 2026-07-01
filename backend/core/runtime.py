"""Shared runtime state for web dashboard and orchestrator loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any


@dataclass(slots=True)
class RuntimeState:
    """Mutable status published through REST and websocket APIs."""

    status: str = "idle"
    paused: bool = False
    device: str = "unknown"
    current_model: str = "bootstrap"
    best_model: str = "bootstrap"
    replay_size: int = 0
    training_step: int = 0
    self_play_games: int = 0
    human_games: int = 0
    human_samples: int = 0
    arena_games: int = 0
    latest_loss: float = 0.0
    latest_elo: float = 0.0
    gpu_util_hint: float = 0.0
    # ---- 训练性能 ----
    train_steps_per_sec: float = 0.0
    games_per_min: float = 0.0
    avg_game_moves: int = 0
    avg_game_ms: float = 0.0
    steps_per_cycle: int = 0
    batch_size: int = 0
    parallel_self_play_games: int = 0
    target_parallel_self_play_games: int = 0
    active_games: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class RuntimeRegistry:
    """Thread-safe storage for global runtime status."""

    def __init__(self) -> None:
        self._state = RuntimeState()
        self._lock = Lock()

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of runtime metrics."""
        with self._lock:
            return asdict(self._state)

    def update(self, **kwargs: Any) -> None:
        """Update one or more fields in runtime state."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
            self._state.updated_at = datetime.now(UTC).isoformat()


runtime_registry = RuntimeRegistry()
