"""Process task entrypoints for self-play games."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from self_play.worker import SelfPlayResult, SelfPlayWorker


@dataclass(slots=True)
class ProcessSelfPlayTask:
    """Serializable per-game task payload for process-based self-play."""

    worker_id: int
    played_games: int
    board_size: int
    win_length: int
    temperature: float
    random_opening_moves: int
    exploration_epsilon: float
    bootstrap_games: int
    heuristic_mix_ratio: float
    prune_keep_ratio: float


def play_one_game_process(task: ProcessSelfPlayTask, conn: Connection | None = None) -> SelfPlayResult:
    """Run one self-play game in a subprocess.

    When a connection is provided, model move requests are proxied to parent
    process where requests can be batched for inference.
    """

    def model_move_proxy(board: np.ndarray, legal_moves: list[int]) -> int | None:
        if conn is None:
            return None
        req: dict[str, Any] = {
            "board": board,
            "legal_moves": legal_moves,
        }
        conn.send(req)
        response = conn.recv()
        if response is None:
            return None
        return int(response)

    worker = SelfPlayWorker(
        board_size=task.board_size,
        win_length=task.win_length,
        model_move_fn=model_move_proxy if conn is not None else None,
        temperature=task.temperature,
        worker_id=task.worker_id,
        progress_cb=None,
        random_opening_moves=task.random_opening_moves,
        exploration_epsilon=task.exploration_epsilon,
        bootstrap_games=task.bootstrap_games,
        heuristic_mix_ratio=task.heuristic_mix_ratio,
        prune_keep_ratio=task.prune_keep_ratio,
    )
    worker._played_games = task.played_games
    result = worker.play_one_game()
    if conn is not None:
        conn.close()
    return result
