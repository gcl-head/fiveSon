"""Self-play workers that generate replay samples using trained model inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from engine.board import Board
from replay_buffer.prioritized import ReplaySample


@dataclass(slots=True)
class SelfPlayResult:
    """Summary for one finished self-play game."""

    moves: int
    winner: int
    samples: list[ReplaySample]


class SelfPlayWorker:
    """Self-play worker using trained model inference with temperature sampling."""

    def __init__(
        self,
        board_size: int,
        win_length: int,
        model_move_fn: Callable[[np.ndarray, list[int]], int | None] | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.board = Board(size=board_size, win_length=win_length)
        self.model_move_fn = model_move_fn
        self.temperature = max(0.01, temperature)
        self.board_size = board_size

    def _select_move(self, board: np.ndarray, legal_moves: list[int]) -> int:
        """Select a move using trained model with temperature sampling, fallback to random if model unavailable."""
        if not legal_moves:
            return -1

        # Try model inference first
        if self.model_move_fn is not None:
            model_move = self.model_move_fn(board, legal_moves)
            if model_move is not None:
                return model_move

        # Fallback: uniform random selection
        return int(np.random.choice(legal_moves))

    def play_one_game(self) -> SelfPlayResult:
        """Play a single game using current trained model inference."""
        state = self.board.initial_state()
        trajectory: list[tuple[np.ndarray, int, int]] = []

        while not self.board.terminal(state):
            legal = self.board.legal_moves(state)
            move = self._select_move(state.board, legal)
            trajectory.append((state.board.copy(), move, state.to_play))
            state = self.board.apply(state, move)

        winner = state.winner
        samples: list[ReplaySample] = []
        size = self.board.size * self.board.size
        for board, move, player in trajectory:
            policy = np.zeros(size, dtype=np.float32)
            policy[move] = 1.0
            value = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
            samples.append(
                ReplaySample(
                    state=board,
                    policy_target=policy,
                    value_target=value,
                    priority=1.0,
                )
            )

        return SelfPlayResult(moves=len(trajectory), winner=winner, samples=samples)
