"""Arena evaluation and best-model promotion policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from engine.board import Board, BoardState
from evaluation.elo import expected_score, update_elo


@dataclass(slots=True)
class ArenaResult:
    """Arena match aggregate metrics."""

    games: int
    challenger_wins: int
    challenger_losses: int
    draws: int
    challenger_win_rate: float
    challenger_elo: float
    best_elo: float
    promoted: bool


class Arena:
    """Evaluate challenger model and decide promotion."""

    def __init__(self, games: int, promotion_win_rate: float, board_size: int, win_length: int) -> None:
        self.games = games
        self.promotion_win_rate = promotion_win_rate
        self.board = Board(size=board_size, win_length=win_length)
        self._rng = np.random.default_rng()

    def _baseline_move(self, state: BoardState, legal: list[int]) -> int:
        center = self.board.size // 2
        center_move = center * self.board.size + center
        if center_move in legal:
            return center_move

        occupied = np.argwhere(state.board != 0)
        if occupied.size == 0:
            return int(self._rng.choice(np.asarray(legal, dtype=np.int64)))

        scored: list[tuple[float, int]] = []
        for move in legal:
            r, c = divmod(move, self.board.size)
            dist = float(np.min(np.abs(occupied[:, 0] - r) + np.abs(occupied[:, 1] - c)))
            scored.append((dist, move))

        scored.sort(key=lambda x: x[0])
        top_k = [m for _, m in scored[: min(6, len(scored))]]
        return int(self._rng.choice(np.asarray(top_k, dtype=np.int64)))

    def _play_one_game(self, challenger_move_fn: Callable[[np.ndarray, list[int]], int | None], challenger_is_black: bool) -> int:
        state = self.board.initial_state()
        while not self.board.terminal(state):
            legal = self.board.legal_moves(state)
            challenger_turn = (state.to_play == 1 and challenger_is_black) or (
                state.to_play == -1 and not challenger_is_black
            )

            if challenger_turn:
                move = challenger_move_fn(state.board, legal)
                if move is None or move not in legal:
                    move = self._baseline_move(state, legal)
            else:
                move = self._baseline_move(state, legal)

            state = self.board.apply(state, int(move))

        if state.winner == 0:
            return 0
        challenger_won = (state.winner == 1 and challenger_is_black) or (
            state.winner == -1 and not challenger_is_black
        )
        return 1 if challenger_won else -1

    def evaluate(self, challenger_move_fn: Callable[[np.ndarray, list[int]], int | None], current_best_elo: float) -> ArenaResult:
        games = max(1, self.games)
        challenger_wins = 0
        challenger_losses = 0
        draws = 0

        for i in range(games):
            result = self._play_one_game(challenger_move_fn=challenger_move_fn, challenger_is_black=(i % 2 == 0))
            if result > 0:
                challenger_wins += 1
            elif result < 0:
                challenger_losses += 1
            else:
                draws += 1

        win_rate = challenger_wins / games

        challenger_elo = current_best_elo
        best_elo = current_best_elo

        actual = (challenger_wins + 0.5 * draws) / games
        exp = expected_score(challenger_elo, best_elo)
        challenger_elo = update_elo(challenger_elo, exp, actual)
        best_elo = update_elo(best_elo, 1 - exp, 1 - actual)

        promoted = win_rate >= self.promotion_win_rate
        return ArenaResult(
            games=games,
            challenger_wins=challenger_wins,
            challenger_losses=challenger_losses,
            draws=draws,
            challenger_win_rate=win_rate,
            challenger_elo=challenger_elo,
            best_elo=best_elo,
            promoted=promoted,
        )
