"""Arena evaluation and best-model promotion policy."""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.elo import expected_score, update_elo


@dataclass(slots=True)
class ArenaResult:
    """Arena match aggregate metrics."""

    games: int
    challenger_win_rate: float
    challenger_elo: float
    best_elo: float
    promoted: bool


class Arena:
    """Evaluate challenger model and decide promotion."""

    def __init__(self, games: int, promotion_win_rate: float) -> None:
        self.games = games
        self.promotion_win_rate = promotion_win_rate

    def evaluate(self, challenger_wins: int, draws: int, current_best_elo: float) -> ArenaResult:
        games = max(1, self.games)
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
            challenger_win_rate=win_rate,
            challenger_elo=challenger_elo,
            best_elo=best_elo,
            promoted=promoted,
        )
