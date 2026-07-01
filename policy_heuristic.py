"""Threat-driven heuristic policy prior for Gomoku.

This module is intentionally lightweight and deterministic enough for engineering
use in bootstrap self-play, MCTS priors, and weak-supervision before a strong
policy/value network is trained.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import exp

BoardGrid = list[list[int]]
Move = tuple[int, int]


@dataclass(frozen=True)
class HeuristicWeights:
    """Scoring weights used by the threat-driven policy prior."""

    win: float = 1_000_000_000.0
    live_four: float = 100_000.0
    rush_four: float = 10_000.0
    live_three: float = 5_000.0
    sleep_three: float = 1_000.0
    live_two: float = 200.0
    adjacency_bonus: float = 50.0
    center_distance_penalty: float = 2.0
    double_live_three_bonus: float = 12_000.0
    defend_live_four_bonus: float = 120_000.0
    defend_win_bonus: float = 250_000.0


class GomokuHeuristicPolicy:
    """Threat-driven heuristic policy scorer for Gomoku move priors.

    Public API:
      - evaluate_board(board, player, move) -> float
      - get_policy(board, player) -> dict[(x, y), prob]

    Board convention:
      - 0: empty
      - 1: player1
      - -1: player2
    """

    _DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))

    def __init__(self, size: int = 15, weights: HeuristicWeights | None = None) -> None:
        self.size = size
        self.weights = weights or HeuristicWeights()

    def evaluate_board(self, board: BoardGrid, player: int, move: Move) -> float:
        """Evaluate one candidate move with threat-aware scoring."""
        x, y = move
        if not self._is_legal(board, x, y):
            return float("-inf")

        score = 0.0
        board_after = self._apply_move(board, player, x, y)

        own = self._analyze_move_threats(board_after, player, x, y)
        opp = self._analyze_move_threats(board_after, -player, x, y)

        if own["win"] > 0:
            return self.weights.win

        score += own["live_four"] * self.weights.live_four
        score += own["rush_four"] * self.weights.rush_four
        score += own["live_three"] * self.weights.live_three
        score += own["sleep_three"] * self.weights.sleep_three
        score += own["live_two"] * self.weights.live_two

        # Reward tactical conversion and pressure creation.
        if own["live_three"] >= 2:
            score += self.weights.double_live_three_bonus

        # Mild defense value when this move also reduces opponent motifs nearby.
        score += opp["live_three"] * 150.0
        score += opp["rush_four"] * 300.0

        # Connectivity: favor moves that connect to nearby friendly stones.
        score += self._connectivity_score(board, player, x, y)

        # Prefer central influence but not overwhelmingly.
        center = self.size // 2
        distance = abs(x - center) + abs(y - center)
        score -= distance * self.weights.center_distance_penalty

        # Encourage local engagement around existing stones.
        if self._has_neighbor(board, x, y, radius=2):
            score += self.weights.adjacency_bonus

        # Hard tactical defense rule: block opponent immediate threats.
        defense_score = self._defense_priority(board, player, x, y)
        score += defense_score

        return score

    def get_policy(self, board: BoardGrid, player: int) -> dict[Move, float]:
        """Return softmax-normalized move prior over all legal points."""
        legal_moves = self._legal_moves(board)
        if not legal_moves:
            return {}

        raw_scores: list[tuple[Move, float]] = []
        for mv in legal_moves:
            raw_scores.append((mv, self.evaluate_board(board, player, mv)))

        # Stable softmax over finite scores.
        finite_scores = [s for _, s in raw_scores if s != float("-inf")]
        if not finite_scores:
            uniform = 1.0 / len(raw_scores)
            return {mv: uniform for mv, _ in raw_scores}

        max_score = max(finite_scores)
        exp_scores: list[tuple[Move, float]] = []
        total = 0.0
        for mv, s in raw_scores:
            # Clamp to keep exp numerically safe in extreme tactical positions.
            e = 0.0 if s == float("-inf") else exp(max(min(s - max_score, 60.0), -60.0))
            exp_scores.append((mv, e))
            total += e

        if total <= 0.0:
            uniform = 1.0 / len(exp_scores)
            return {mv: uniform for mv, _ in exp_scores}

        return {mv: val / total for mv, val in exp_scores}

    def _defense_priority(self, board: BoardGrid, player: int, x: int, y: int) -> float:
        """Elevate moves that must block opponent's immediate tactical threats."""
        opp = -player
        defense_bonus = 0.0

        opp_win_moves = self._find_immediate_pattern_moves(board, opp, target="win")
        opp_live_four_moves = self._find_immediate_pattern_moves(board, opp, target="live_four")

        if (x, y) in opp_win_moves:
            defense_bonus += self.weights.defend_win_bonus
        if (x, y) in opp_live_four_moves:
            defense_bonus += self.weights.defend_live_four_bonus

        return defense_bonus

    def _find_immediate_pattern_moves(self, board: BoardGrid, player: int, target: str) -> set[Move]:
        """Find legal moves that instantly create target pattern for given player."""
        moves: set[Move] = set()
        for x, y in self._legal_moves(board):
            b2 = self._apply_move(board, player, x, y)
            motifs = self._analyze_move_threats(b2, player, x, y)
            if (target == "win" and motifs["win"] > 0) or (target == "live_four" and motifs["live_four"] > 0):
                moves.add((x, y))
        return moves

    def _analyze_move_threats(self, board: BoardGrid, player: int, x: int, y: int) -> dict[str, int]:
        """Detect core threat motifs generated by a stone at (x, y)."""
        if board[x][y] != player:
            return {
                "win": 0,
                "live_four": 0,
                "rush_four": 0,
                "live_three": 0,
                "sleep_three": 0,
                "live_two": 0,
            }

        motifs = {
            "win": 0,
            "live_four": 0,
            "rush_four": 0,
            "live_three": 0,
            "sleep_three": 0,
            "live_two": 0,
        }

        for dx, dy in self._DIRECTIONS:
            left_count, left_open = self._count_in_direction(board, player, x, y, -dx, -dy)
            right_count, right_open = self._count_in_direction(board, player, x, y, dx, dy)

            total = 1 + left_count + right_count
            open_ends = int(left_open) + int(right_open)

            if total >= 5:
                motifs["win"] += 1
                continue

            if total == 4 and open_ends == 2:
                motifs["live_four"] += 1
            elif total == 4 and open_ends == 1:
                motifs["rush_four"] += 1
            elif total == 3 and open_ends == 2:
                motifs["live_three"] += 1
            elif total == 3 and open_ends == 1:
                motifs["sleep_three"] += 1
            elif total == 2 and open_ends == 2:
                motifs["live_two"] += 1

        return motifs

    def _count_in_direction(
        self,
        board: BoardGrid,
        player: int,
        x: int,
        y: int,
        dx: int,
        dy: int,
    ) -> tuple[int, bool]:
        count = 0
        cx = x + dx
        cy = y + dy
        while self._in_bounds(cx, cy) and board[cx][cy] == player:
            count += 1
            cx += dx
            cy += dy
        open_end = self._in_bounds(cx, cy) and board[cx][cy] == 0
        return count, open_end

    def _connectivity_score(self, board: BoardGrid, player: int, x: int, y: int) -> float:
        """Reward nearby friendly groups; closer neighbors score higher."""
        score = 0.0
        for nx, ny in self._iter_neighbors(x, y, radius=2):
            if not self._in_bounds(nx, ny):
                continue
            if board[nx][ny] != player:
                continue
            dist = abs(nx - x) + abs(ny - y)
            if dist == 0:
                continue
            score += self.weights.adjacency_bonus / float(dist)
        return score

    def _has_neighbor(self, board: BoardGrid, x: int, y: int, radius: int = 2) -> bool:
        for nx, ny in self._iter_neighbors(x, y, radius):
            if self._in_bounds(nx, ny) and board[nx][ny] != 0:
                return True
        return False

    def _iter_neighbors(self, x: int, y: int, radius: int) -> Iterable[Move]:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                yield x + dx, y + dy

    def _legal_moves(self, board: BoardGrid) -> list[Move]:
        moves: list[Move] = []
        for x in range(self.size):
            for y in range(self.size):
                if board[x][y] == 0:
                    moves.append((x, y))
        return moves

    def _is_legal(self, board: BoardGrid, x: int, y: int) -> bool:
        return self._in_bounds(x, y) and board[x][y] == 0

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    @staticmethod
    def _apply_move(board: BoardGrid, player: int, x: int, y: int) -> BoardGrid:
        copied = [row[:] for row in board]
        copied[x][y] = player
        return copied
