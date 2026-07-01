"""In-memory game session service for web UI human-vs-AI interactions."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from backend.core.runtime import runtime_registry
from backend.services.training_bridge import training_bridge
from engine.board import Board, BoardState
from self_play.policy_heuristic import GomokuHeuristicPolicy


@dataclass(slots=True)
class MoveResult:
    """Result payload for a full player turn and optional AI response."""

    board: list[list[int]]
    to_play: int
    winner: int
    ai_move: int | None
    legal_moves: int
    message: str


@dataclass(slots=True)
class QuickEvalResult:
    """Aggregated result for quick online validation matches."""

    games: int
    generation: int
    baseline_generation: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_moves: float
    avg_game_ms: float


class GameService:
    """Stateful game manager shared by API handlers."""

    def __init__(self, size: int = 15, win_length: int = 5) -> None:
        self._board = Board(size=size, win_length=win_length)
        self._heuristic = GomokuHeuristicPolicy(size=size)
        self._state: BoardState = self._board.initial_state()
        self._trajectory: list[tuple[np.ndarray, int, int]] = []

    def state(self) -> dict[str, object]:
        legal = self._board.legal_moves(self._state)
        generation = training_bridge.deployed_generation()
        model_move = training_bridge.select_model_move(self._state.board, legal)
        return {
            "size": self._board.size,
            "board": self._state.board.tolist(),
            "to_play": self._state.to_play,
            "winner": self._state.winner,
            "move_count": self._state.move_count,
            "legal_moves": len(legal),
            "deployed_generation": generation,
            "ai_policy": "trained_model" if model_move is not None else "heuristic_policy",
        }

    def reset(self) -> dict[str, object]:
        self._state = self._board.initial_state()
        self._trajectory = []
        return self.state()

    def play_human_move(self, move: int) -> MoveResult:
        if self._state.winner != 0:
            return MoveResult(
                board=self._state.board.tolist(),
                to_play=self._state.to_play,
                winner=self._state.winner,
                ai_move=None,
                legal_moves=0,
                message="game already finished",
            )

        self._trajectory.append((self._state.board.copy(), move, self._state.to_play))
        self._state = self._board.apply(self._state, move)
        if self._board.terminal(self._state):
            self._ingest_finished_game()
            return MoveResult(
                board=self._state.board.tolist(),
                to_play=self._state.to_play,
                winner=self._state.winner,
                ai_move=None,
                legal_moves=len(self._board.legal_moves(self._state)),
                message="human move finished the game",
            )

        ai_move = self._pick_ai_move(self._state)
        self._trajectory.append((self._state.board.copy(), ai_move, self._state.to_play))
        self._state = self._board.apply(self._state, ai_move)

        message = "ai moved"
        if self._state.winner != 0:
            message = "ai won"
        elif self._board.terminal(self._state):
            message = "draw"

        if self._board.terminal(self._state):
            self._ingest_finished_game()

        return MoveResult(
            board=self._state.board.tolist(),
            to_play=self._state.to_play,
            winner=self._state.winner,
            ai_move=ai_move,
            legal_moves=len(self._board.legal_moves(self._state)),
            message=message,
        )

    def _pick_ai_move_with_generation(self, state: BoardState, generation: int, use_model: bool = True) -> int:
        legal = self._board.legal_moves(state)
        if use_model:
            model_move = training_bridge.select_model_move(state.board, legal)
            if model_move is not None:
                return model_move

        heuristic_policy = self._heuristic.get_policy(state.board.tolist(), state.to_play)
        best_move = None
        best_prob = -1.0
        for move in legal:
            x, y = divmod(move, self._board.size)
            prob = float(heuristic_policy.get((x, y), 0.0))
            if prob > best_prob:
                best_prob = prob
                best_move = move

        if best_move is not None:
            return int(best_move)
        return int(np.random.choice(legal))

    def _pick_ai_move(self, state: BoardState) -> int:
        generation = training_bridge.deployed_generation()
        return self._pick_ai_move_with_generation(state, generation)

    def quick_eval(self, games: int = 30, baseline_generation: int = 0) -> QuickEvalResult:
        """Evaluate deployed generation against a fixed baseline in fast simulated games."""
        games = max(2, min(games, 200))
        generation = training_bridge.deployed_generation()
        wins = 0
        losses = 0
        draws = 0
        total_moves = 0
        t0 = time.perf_counter()

        for i in range(games):
            state = self._board.initial_state()
            challenger_is_black = (i % 2 == 0)

            while not self._board.terminal(state):
                challenger_turn = (state.to_play == 1 and challenger_is_black) or (
                    state.to_play == -1 and not challenger_is_black
                )
                move = self._pick_ai_move_with_generation(
                    state,
                    generation if challenger_turn else baseline_generation,
                    use_model=challenger_turn,
                )
                state = self._board.apply(state, move)

            total_moves += state.move_count
            if state.winner == 0:
                draws += 1
            else:
                challenger_won = (state.winner == 1 and challenger_is_black) or (
                    state.winner == -1 and not challenger_is_black
                )
                if challenger_won:
                    wins += 1
                else:
                    losses += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return QuickEvalResult(
            games=games,
            generation=generation,
            baseline_generation=baseline_generation,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=(wins / games) if games else 0.0,
            avg_moves=(total_moves / games) if games else 0.0,
            avg_game_ms=(elapsed_ms / games) if games else 0.0,
        )

    def _ingest_finished_game(self) -> None:
        inserted = training_bridge.ingest_human_game(
            trajectory=self._trajectory,
            winner=self._state.winner,
            board_size=self._board.size,
        )
        if inserted > 0:
            snapshot = runtime_registry.snapshot()
            runtime_registry.update(
                replay_size=int(snapshot.get("replay_size", 0)) + inserted,
                human_games=int(snapshot.get("human_games", 0)) + 1,
                human_samples=int(snapshot.get("human_samples", 0)) + inserted,
            )
        self._trajectory = []


game_service = GameService()
