"""Gomoku board representation and move legality checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Player = int


@dataclass(slots=True)
class BoardState:
    """Immutable-like board state used by MCTS and self-play."""

    board: NDArray[np.int8]
    to_play: Player
    move_count: int
    winner: Player


class Board:
    """Board utility with win detection and legal move generation."""

    def __init__(self, size: int = 15, win_length: int = 5) -> None:
        self.size = size
        self.win_length = win_length

    def initial_state(self) -> BoardState:
        return BoardState(
            board=np.zeros((self.size, self.size), dtype=np.int8),
            to_play=1,
            move_count=0,
            winner=0,
        )

    def legal_moves(self, state: BoardState) -> list[int]:
        coords = np.argwhere(state.board == 0)
        return [int(r * self.size + c) for r, c in coords]

    def apply(self, state: BoardState, move: int) -> BoardState:
        r, c = divmod(move, self.size)
        if state.board[r, c] != 0 or state.winner != 0:
            raise ValueError("invalid move")

        next_board = state.board.copy()
        next_board[r, c] = state.to_play
        winner = self._detect_winner(next_board, r, c)
        return BoardState(
            board=next_board,
            to_play=-state.to_play,
            move_count=state.move_count + 1,
            winner=winner,
        )

    def terminal(self, state: BoardState) -> bool:
        return state.winner != 0 or state.move_count >= self.size * self.size

    def _detect_winner(self, board: NDArray[np.int8], row: int, col: int) -> Player:
        player = int(board[row, col])
        if player == 0:
            return 0

        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in dirs:
            count = 1
            count += self._count(board, row, col, dr, dc, player)
            count += self._count(board, row, col, -dr, -dc, player)
            if count >= self.win_length:
                return player
        return 0

    def _count(
        self,
        board: NDArray[np.int8],
        row: int,
        col: int,
        dr: int,
        dc: int,
        player: Player,
    ) -> int:
        total = 0
        r = row + dr
        c = col + dc
        while 0 <= r < self.size and 0 <= c < self.size and int(board[r, c]) == player:
            total += 1
            r += dr
            c += dc
        return total
