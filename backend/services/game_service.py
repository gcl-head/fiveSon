"""In-memory game session service for web UI human-vs-AI interactions."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from threading import Lock

import numpy as np

from backend.core.runtime import runtime_registry
from backend.services.training_bridge import training_bridge
from engine.board import Board, BoardState
from self_play.policy_heuristic import GomokuHeuristicPolicy
from replay_buffer.prioritized import PrioritizedReplayBuffer
from training.trainer import Trainer


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


@dataclass(slots=True)
class QuickEvalGameState:
    """Live quick-eval snapshot published to the dashboard."""

    game_id: int
    board: list[list[int]]
    move_count: int
    winner: int
    done: bool
    elapsed_ms: float
    generation: int
    baseline_generation: int


class GameService:
    """Stateful game manager shared by API handlers."""

    def __init__(self, size: int = 15, win_length: int = 5) -> None:
        self._board = Board(size=size, win_length=win_length)
        self._heuristic = GomokuHeuristicPolicy(size=size)
        self._state: BoardState = self._board.initial_state()
        self._trajectory: list[tuple[np.ndarray, int, int]] = []
        self._checkpoint_dir = Path(__file__).resolve().parents[2] / "checkpoints"
        self._quick_eval_models: dict[int, Callable[[np.ndarray, list[int]], int | None]] = {}
        self._quick_eval_games: dict[int, QuickEvalGameState] = {}
        self._quick_eval_lock = Lock()

    def _publish_quick_eval_game(self, game: QuickEvalGameState) -> None:
        with self._quick_eval_lock:
            self._quick_eval_games[game.game_id] = game
            payload = [
                {
                    "game_id": item.game_id,
                    "board": item.board,
                    "move_count": item.move_count,
                    "winner": item.winner,
                    "done": item.done,
                    "elapsed_ms": round(item.elapsed_ms, 1),
                    "generation": item.generation,
                    "baseline_generation": item.baseline_generation,
                }
                for item in sorted(self._quick_eval_games.values(), key=lambda value: value.game_id)
            ]
        runtime_registry.update(quick_eval_games=payload)

    def _clear_quick_eval_games(self) -> None:
        with self._quick_eval_lock:
            self._quick_eval_games = {}
        runtime_registry.update(quick_eval_games=[])

    def _cache_generation_model(self, generation: int) -> Callable[[np.ndarray, list[int]], int | None] | None:
        if generation <= 0:
            return None

        with self._quick_eval_lock:
            cached = self._quick_eval_models.get(generation)
        if cached is not None:
            return cached

        replay = PrioritizedReplayBuffer(capacity=1, prioritized=False, alpha=0.6)
        trainer = Trainer(
            replay,
            board_size=self._board.size,
            batch_size=1,
            device="cpu",
            amp_enabled=False,
            learning_rate=0.001,
        )
        meta = trainer.load_generation_checkpoint(
            self._checkpoint_dir,
            generation=generation,
            board_size=self._board.size,
            action_dim=self._board.size * self._board.size,
        )
        if meta is None:
            return None

        move_fn = trainer.infer_move
        with self._quick_eval_lock:
            self._quick_eval_models[generation] = move_fn
        return move_fn

    def _pick_move_for_generation(
        self,
        state: BoardState,
        generation: int,
        move_fn: Callable[[np.ndarray, list[int]], int | None] | None,
    ) -> int:
        legal = self._board.legal_moves(state)
        if generation > 0 and move_fn is not None:
            move = move_fn(state.board, legal)
            if move is not None and move in legal:
                return int(move)
        return self._pick_stochastic_heuristic_move(state)

    def _play_quick_eval_game(
        self,
        game_id: int,
        generation: int,
        baseline_generation: int,
        challenger_move_fn: Callable[[np.ndarray, list[int]], int | None] | None,
        baseline_move_fn: Callable[[np.ndarray, list[int]], int | None] | None,
    ) -> tuple[QuickEvalGameState, int, int, int, int, float]:
        t0 = time.perf_counter()
        state = self._board.initial_state()
        challenger_is_black = (game_id % 2 == 0)
        snapshot = QuickEvalGameState(
            game_id=game_id,
            board=state.board.tolist(),
            move_count=state.move_count,
            winner=state.winner,
            done=False,
            elapsed_ms=0.0,
            generation=generation,
            baseline_generation=baseline_generation,
        )
        self._publish_quick_eval_game(snapshot)

        opening_noise = min(2, len(self._board.legal_moves(state)))
        for _ in range(opening_noise):
            legal = self._board.legal_moves(state)
            if not legal or self._board.terminal(state):
                break
            state = self._board.apply(state, int(np.random.choice(legal)))

        wins = 0
        losses = 0
        draws = 0
        total_moves = 0
        while not self._board.terminal(state):
            challenger_turn = (state.to_play == 1 and challenger_is_black) or (
                state.to_play == -1 and not challenger_is_black
            )
            if challenger_turn:
                move = self._pick_move_for_generation(state, generation, challenger_move_fn)
            else:
                move = self._pick_move_for_generation(state, baseline_generation, baseline_move_fn)
            state = self._board.apply(state, move)
            snapshot = QuickEvalGameState(
                game_id=game_id,
                board=state.board.tolist(),
                move_count=state.move_count,
                winner=state.winner,
                done=self._board.terminal(state),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                generation=generation,
                baseline_generation=baseline_generation,
            )
            self._publish_quick_eval_game(snapshot)

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

        final_snapshot = QuickEvalGameState(
            game_id=game_id,
            board=state.board.tolist(),
            move_count=state.move_count,
            winner=state.winner,
            done=True,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            generation=generation,
            baseline_generation=baseline_generation,
        )
        self._publish_quick_eval_game(final_snapshot)
        return final_snapshot, wins, losses, draws, total_moves, final_snapshot.elapsed_ms

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
        if use_model and generation > 0:
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

    def _pick_stochastic_heuristic_move(self, state: BoardState) -> int:
        legal = self._board.legal_moves(state)
        if not legal:
            return -1
        heuristic_policy = self._heuristic.get_policy(state.board.tolist(), state.to_play)
        probs = np.array([
            max(0.0, float(heuristic_policy.get(divmod(move, self._board.size), 0.0))) for move in legal
        ], dtype=np.float64)
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0:
            return int(np.random.choice(legal))
        probs /= prob_sum
        idx = int(np.random.choice(np.arange(len(legal)), p=probs))
        return int(legal[idx])

    def _pick_ai_move(self, state: BoardState) -> int:
        generation = training_bridge.deployed_generation()
        return self._pick_ai_move_with_generation(state, generation)

    def quick_eval(
        self,
        games: int = 30,
        generation: int | None = None,
        baseline_generation: int = 0,
    ) -> QuickEvalResult:
        """Evaluate deployed generation against a fixed baseline in fast simulated games."""
        games = max(2, min(games, 200))
        generation = training_bridge.deployed_generation() if generation is None else int(generation)
        deployed_generation = training_bridge.deployed_generation()
        if generation < 0 or generation > deployed_generation:
            raise ValueError(f"generation {generation} is out of range 0..{deployed_generation}")
        if baseline_generation < 0 or baseline_generation > deployed_generation:
            raise ValueError(f"baseline generation {baseline_generation} is out of range 0..{deployed_generation}")

        challenger_move_fn = None if generation == 0 else training_bridge.select_model_move
        if generation > 0:
            cached_move_fn = self._cache_generation_model(generation)
            if cached_move_fn is None:
                raise ValueError(f"generation {generation} checkpoint not found")
            challenger_move_fn = cached_move_fn

        baseline_move_fn = None
        if baseline_generation > 0:
            baseline_move_fn = self._cache_generation_model(baseline_generation)
            if baseline_move_fn is None:
                raise ValueError(f"generation {baseline_generation} checkpoint not found")

        wins = 0
        losses = 0
        draws = 0
        total_moves = 0
        t0 = time.perf_counter()
        self._clear_quick_eval_games()

        def run_one(game_id: int) -> tuple[int, int, int, int]:
            snapshot, game_wins, game_losses, game_draws, game_moves, _ = self._play_quick_eval_game(
                game_id=game_id,
                generation=generation,
                baseline_generation=baseline_generation,
                challenger_move_fn=challenger_move_fn,
                baseline_move_fn=baseline_move_fn,
            )
            del snapshot
            return game_wins, game_losses, game_draws, game_moves

        max_workers = min(32, games)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, game_id) for game_id in range(games)]
            for future in as_completed(futures):
                game_wins, game_losses, game_draws, game_moves = future.result()
                wins += game_wins
                losses += game_losses
                draws += game_draws
                total_moves += game_moves

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._clear_quick_eval_games()
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
