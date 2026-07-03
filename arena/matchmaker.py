"""Arena evaluation and best-model promotion policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from engine.board import Board, BoardState
from evaluation.elo import expected_score, update_elo
from self_play.policy_heuristic import GomokuHeuristicPolicy


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
        self.heuristic = GomokuHeuristicPolicy(size=board_size)
        self._rng = np.random.default_rng()

    def _baseline_move(self, state: BoardState, legal: list[int]) -> int:
        if not legal:
            return -1

        heuristic_policy = self.heuristic.get_policy(state.board.tolist(), state.to_play)
        probs = np.array([
            max(0.0, float(heuristic_policy.get(divmod(move, self.board.size), 0.0))) for move in legal
        ], dtype=np.float64)
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0:
            return int(self._rng.choice(np.asarray(legal, dtype=np.int64)))
        probs /= prob_sum
        idx = int(self._rng.choice(np.arange(len(legal)), p=probs))
        return int(legal[idx])

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
