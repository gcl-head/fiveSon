"""Self-play workers that generate replay samples using trained model inference."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from engine.board import Board
from replay_buffer.prioritized import ReplaySample
from self_play.policy_heuristic import GomokuHeuristicPolicy


@dataclass(slots=True)
class SelfPlayResult:
    """Summary for one finished self-play game."""

    moves: int
    winner: int
    elapsed_ms: float
    heuristic_moves: int
    model_moves: int
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
        bootstrap_games: int = 80,
        heuristic_mix_ratio: float = 0.25,
        prune_keep_ratio: float = 0.6,
    ) -> None:
        self.board = Board(size=board_size, win_length=win_length)
        self.heuristic = GomokuHeuristicPolicy(size=board_size)
        self.model_move_fn = model_move_fn
        self.temperature = max(0.01, temperature)
        self.board_size = board_size
        self.worker_id = worker_id
        self.progress_cb = progress_cb
        self.random_opening_moves = max(0, random_opening_moves)
        self.exploration_epsilon = float(max(0.0, min(1.0, exploration_epsilon)))
        self.bootstrap_games = max(0, bootstrap_games)
        self.heuristic_mix_ratio = float(max(0.0, min(1.0, heuristic_mix_ratio)))
        self.prune_keep_ratio = float(max(0.2, min(1.0, prune_keep_ratio)))
        self._played_games = 0
        # Worker-local RNG avoids all parallel games sharing identical random sequence.
        self.rng = np.random.default_rng(seed=int(time.time_ns() % (2**32)) + worker_id * 9973)

    def _move_to_xy(self, move: int) -> tuple[int, int]:
        return divmod(move, self.board_size)

    def _xy_to_move(self, x: int, y: int) -> int:
        return x * self.board_size + y

    def _heuristic_prior(self, board: np.ndarray, player: int, legal_moves: list[int]) -> np.ndarray:
        total = self.board_size * self.board_size
        prior = np.zeros(total, dtype=np.float32)
        if not legal_moves:
            return prior

        board_list = board.tolist()
        occupied = int(np.count_nonzero(board))

        # Evaluate only tactical candidates for speed; this keeps self-play throughput practical.
        if occupied == 0:
            center = self.board_size // 2
            candidates = [
                self._xy_to_move(x, y)
                for x in range(max(0, center - 2), min(self.board_size, center + 3))
                for y in range(max(0, center - 2), min(self.board_size, center + 3))
                if self._xy_to_move(x, y) in legal_moves
            ]
        else:
            cand_set: set[int] = set()
            stones = np.argwhere(board != 0)
            for rx, ry in stones:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        x = int(rx + dx)
                        y = int(ry + dy)
                        if 0 <= x < self.board_size and 0 <= y < self.board_size:
                            mv = self._xy_to_move(x, y)
                            if mv in legal_moves:
                                cand_set.add(mv)
            candidates = list(cand_set)

        if not candidates:
            candidates = legal_moves

        scores: list[tuple[int, float]] = []
        for mv in candidates:
            x, y = self._move_to_xy(mv)
            score = self.heuristic.evaluate_board(board_list, player, (x, y))
            scores.append((mv, score))

        max_score = max(s for _, s in scores)
        total_prob = 0.0
        for mv, score in scores:
            p = float(np.exp(max(min(score - max_score, 40.0), -40.0)))
            prior[mv] = p
            total_prob += p

        # Keep tiny exploration mass on non-candidate legal moves.
        residual = [mv for mv in legal_moves if mv not in {m for m, _ in scores}]
        if residual:
            residual_mass = max(1e-3, total_prob * 0.02)
            each = residual_mass / len(residual)
            for mv in residual:
                prior[mv] = float(each)

        s = float(prior.sum())
        if s <= 0.0:
            p = 1.0 / float(len(legal_moves))
            for mv in legal_moves:
                prior[mv] = p
            return prior
        prior /= s
        return prior

    def _prune_legal_by_prior(self, legal_moves: list[int], prior: np.ndarray) -> list[int]:
        if len(legal_moves) <= 8:
            return legal_moves
        keep_n = max(8, int(len(legal_moves) * self.prune_keep_ratio))
        ranked = sorted(legal_moves, key=lambda mv: float(prior[mv]), reverse=True)
        return ranked[:keep_n]

    def _sample_by_prior(self, moves: list[int], prior: np.ndarray) -> int:
        probs = np.array([float(prior[mv]) for mv in moves], dtype=np.float64)
        s = float(probs.sum())
        if s <= 0.0:
            return int(self.rng.choice(np.asarray(moves, dtype=np.int64)))
        probs /= s
        idx = int(self.rng.choice(np.arange(len(moves)), p=probs))
        return int(moves[idx])

    def _build_target(self, move: int, heuristic_prior: np.ndarray, use_blend: bool) -> np.ndarray:
        total = self.board_size * self.board_size
        one_hot = np.zeros(total, dtype=np.float32)
        one_hot[move] = 1.0
        if not use_blend:
            return one_hot

        mixed = (1.0 - self.heuristic_mix_ratio) * one_hot + self.heuristic_mix_ratio * heuristic_prior
        denom = float(mixed.sum())
        if denom > 0.0:
            mixed /= denom
        return mixed.astype(np.float32)

    def _select_move(
        self,
        board: np.ndarray,
        player: int,
        legal_moves: list[int],
        move_count: int,
    ) -> tuple[int, np.ndarray, str]:
        """Select move and emit policy target for replay training."""
        if not legal_moves:
            total = self.board_size * self.board_size
            return -1, np.zeros(total, dtype=np.float32), "heuristic"

        heuristic_prior = self._heuristic_prior(board, player, legal_moves)
        pruned_moves = self._prune_legal_by_prior(legal_moves, heuristic_prior)

        force_heuristic = (
            self.model_move_fn is None
            or self._played_games < self.bootstrap_games
            or move_count < self.random_opening_moves
        )

        if force_heuristic or float(self.rng.random()) < self.exploration_epsilon:
            move = self._sample_by_prior(pruned_moves, heuristic_prior)
            return move, self._build_target(move, heuristic_prior, use_blend=True), "heuristic"

        model_move = self.model_move_fn(board, pruned_moves) if self.model_move_fn is not None else None
        if model_move is None or model_move not in pruned_moves:
            move = self._sample_by_prior(pruned_moves, heuristic_prior)
            return move, self._build_target(move, heuristic_prior, use_blend=True), "heuristic"

        return int(model_move), self._build_target(int(model_move), heuristic_prior, use_blend=True), "model"

    def play_one_game(self) -> SelfPlayResult:
        """Play a single game using current trained model inference."""
        t0 = time.perf_counter()
        state = self.board.initial_state()
        trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []
        heuristic_moves = 0
        model_moves = 0

        if self.progress_cb is not None:
            self.progress_cb(self.worker_id, state.board.copy(), 0, state.winner, False, 0.0)

        while not self.board.terminal(state):
            legal = self.board.legal_moves(state)
            move, policy_target, source = self._select_move(state.board, state.to_play, legal, state.move_count)
            trajectory.append((state.board.copy(), policy_target, state.to_play))
            if source == "model":
                model_moves += 1
            else:
                heuristic_moves += 1
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
        for board, policy, player in trajectory:
            value = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
            samples.append(
                ReplaySample(
                    state=board,
                    policy_target=policy,
                    value_target=value,
                    priority=1.0,
                )
            )

        self._played_games += 1

        return SelfPlayResult(
            moves=len(trajectory),
            winner=winner,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            heuristic_moves=heuristic_moves,
            model_moves=model_moves,
            samples=samples,
        )
