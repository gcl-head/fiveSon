"""Self-play workers that generate replay samples using trained model inference."""

from __future__ import annotations

import time
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
    elapsed_ms: float
    samples: list[ReplaySample]


class SelfPlayWorker:
    """Self-play worker using trained model inference with temperature sampling."""

    def __init__(
        self,
        board_size: int,
        win_length: int,
        model_move_fn: Callable[[np.ndarray, list[int]], int | None] | None = None,
        temperature: float = 1.0,
        worker_id: int = 0,
        progress_cb: Callable[[int, np.ndarray, int, int, bool, float], None] | None = None,
        random_opening_moves: int = 2,
        exploration_epsilon: float = 0.08,
    ) -> None:
        self.board = Board(size=board_size, win_length=win_length)
        self.model_move_fn = model_move_fn
        self.temperature = max(0.01, temperature)
        self.board_size = board_size
        self.worker_id = worker_id
        self.progress_cb = progress_cb
        self.random_opening_moves = max(0, random_opening_moves)
        self.exploration_epsilon = float(max(0.0, min(1.0, exploration_epsilon)))
        # Worker-local RNG avoids all parallel games sharing identical random sequence.
        self.rng = np.random.default_rng(seed=int(time.time_ns() % (2**32)) + worker_id * 9973)

    def _select_move(self, board: np.ndarray, legal_moves: list[int], move_count: int) -> int:
        """Select a move using trained model with temperature sampling, fallback to random if model unavailable."""
        if not legal_moves:
            return -1

        # Force exploration in early opening so parallel workers quickly diverge.
        if move_count < self.random_opening_moves:
            return int(self.rng.choice(np.asarray(legal_moves, dtype=np.int64)))

        # Try model inference first
        if self.model_move_fn is not None:
            if self.exploration_epsilon > 0 and float(self.rng.random()) < self.exploration_epsilon:
                return int(self.rng.choice(np.asarray(legal_moves, dtype=np.int64)))
            model_move = self.model_move_fn(board, legal_moves)
            if model_move is not None:
                return model_move

        # Fallback: uniform random selection
        return int(self.rng.choice(np.asarray(legal_moves, dtype=np.int64)))

    def play_one_game(self) -> SelfPlayResult:
        """Play a single game using current trained model inference."""
        t0 = time.perf_counter()
        state = self.board.initial_state()
        trajectory: list[tuple[np.ndarray, int, int]] = []

        if self.progress_cb is not None:
            self.progress_cb(self.worker_id, state.board.copy(), 0, state.winner, False, 0.0)

        while not self.board.terminal(state):
            legal = self.board.legal_moves(state)
            move = self._select_move(state.board, legal, state.move_count)
            trajectory.append((state.board.copy(), move, state.to_play))
            state = self.board.apply(state, move)
            if self.progress_cb is not None:
                self.progress_cb(
                    self.worker_id,
                    state.board.copy(),
                    state.move_count,
                    state.winner,
                    self.board.terminal(state),
                    (time.perf_counter() - t0) * 1000.0,
                )

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

        return SelfPlayResult(
            moves=len(trajectory),
            winner=winner,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            samples=samples,
        )
