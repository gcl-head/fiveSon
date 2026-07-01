"""Bridge between orchestrator training loop and web gameplay service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from replay_buffer.prioritized import PrioritizedReplayBuffer, ReplaySample


@dataclass(slots=True)
class BridgeConfig:
    """Tunable bridge options loaded from YAML."""

    model_switch_interval_steps: int = 500
    human_game_priority_weight: float = 8.0


class TrainingBridge:
    """Shared in-process coordination for model switch and human sample ingestion."""

    def __init__(self) -> None:
        self._config = BridgeConfig()
        self._replay: PrioritizedReplayBuffer | None = None
        self._generation = 0
        self._model_move_fn: Callable[[np.ndarray, list[int]], int | None] | None = None
        self._lock = Lock()

    def configure(self, raw_cfg: dict[str, Any]) -> None:
        """Load bridge config from YAML dictionary."""
        webplay = raw_cfg.get("webplay", {})
        with self._lock:
            self._config = BridgeConfig(
                model_switch_interval_steps=max(1, int(webplay.get("model_switch_interval_steps", 500))),
                human_game_priority_weight=max(1.0, float(webplay.get("human_game_priority_weight", 8.0))),
            )

    def attach_replay_buffer(self, replay: PrioritizedReplayBuffer) -> None:
        """Attach replay buffer for cross-service training sample ingestion."""
        with self._lock:
            self._replay = replay

    def register_model_move_fn(self, move_fn: Callable[[np.ndarray, list[int]], int | None]) -> None:
        """Register trained-model move function for online gameplay inference."""
        with self._lock:
            self._model_move_fn = move_fn

    def select_model_move(self, board: np.ndarray, legal_moves: list[int]) -> int | None:
        """Try selecting a move from current trained model callback."""
        with self._lock:
            move_fn = self._model_move_fn

        if move_fn is None:
            return None
        return move_fn(board, legal_moves)

    def maybe_switch_model(self, training_step: int) -> str | None:
        """Return new model name when step crosses configured interval."""
        with self._lock:
            interval = self._config.model_switch_interval_steps
            generation = training_step // interval
            if generation <= self._generation:
                return None

            self._generation = generation
            return f"checkpoint-g{generation}-s{training_step}"

    def deployed_generation(self) -> int:
        """Get currently deployed gameplay generation."""
        with self._lock:
            return self._generation

    def ingest_human_game(
        self,
        trajectory: list[tuple[np.ndarray, int, int]],
        winner: int,
        board_size: int,
    ) -> int:
        """Push human game trajectory into replay buffer with higher sampling priority."""
        with self._lock:
            replay = self._replay
            priority = self._config.human_game_priority_weight

        if replay is None or not trajectory:
            return 0

        total = board_size * board_size
        inserted = 0
        for board, move, player in trajectory:
            policy = np.zeros(total, dtype=np.float32)
            policy[move] = 1.0
            value = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
            replay.push(
                ReplaySample(
                    state=board.copy(),
                    policy_target=policy,
                    value_target=value,
                    priority=priority,
                )
            )
            inserted += 1

        return inserted


training_bridge = TrainingBridge()
