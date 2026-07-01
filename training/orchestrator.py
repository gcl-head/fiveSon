"""Forever orchestrator: self-play -> replay -> train -> evaluate -> promote."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Lock

import numpy as np

from arena.matchmaker import Arena
from backend.core.runtime import runtime_registry
from backend.core.settings import choose_device_config, load_app_config, load_yaml
from backend.services.training_bridge import training_bridge
from replay_buffer.prioritized import PrioritizedReplayBuffer
from self_play.process_worker import ProcessSelfPlayTask, play_one_game_process
from self_play.worker import SelfPlayResult
from training.trainer import Trainer


class Orchestrator:
    """Coordinates long-running loops with runtime status updates."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.app_cfg = load_app_config(config_path)
        self.raw_cfg = load_yaml(config_path)
        self.device_cfg = choose_device_config(config_path)

        replay_cfg = self.raw_cfg["replay_buffer"]
        self.replay = PrioritizedReplayBuffer(
            capacity=int(replay_cfg["capacity"]),
            prioritized=bool(replay_cfg["prioritized"]),
            alpha=float(replay_cfg["alpha"]),
        )
        training_bridge.configure(self.raw_cfg)
        training_bridge.attach_replay_buffer(self.replay)

        self._parallel_self_play_games = max(
            8,
            int(self.raw_cfg.get("self_play", {}).get("parallel_games", self.raw_cfg.get("self_play", {}).get("workers", 8))),
        )
        self._active_games: dict[int, dict[str, object]] = {}
        self._active_games_lock = Lock()
        self._self_play_games_completed = 0
        self._self_play_pool: ProcessPoolExecutor | None = None
        self.trainer = Trainer(
            self.replay,
            batch_size=self.device_cfg.batch_size,
            device=self.device_cfg.device,
            amp_enabled=self.device_cfg.amp_enabled,
            learning_rate=float(self.raw_cfg["training"].get("learning_rate", 0.001)),
        )
        # Wire model inference to both bridge (for gameplay) and worker (for self-play data generation)
        training_bridge.register_model_move_fn(self.trainer.infer_move)
        self._set_parallel_self_play_games(self._parallel_self_play_games)
        self.arena = Arena(
            games=self.app_cfg.arena_games,
            promotion_win_rate=self.app_cfg.promotion_win_rate,
            board_size=self.app_cfg.board_size,
            win_length=self.app_cfg.win_length,
        )

        runtime_registry.update(status="boot", device=self.device_cfg.device)
        self._steps_per_cycle = int(self.raw_cfg["training"].get("steps_per_cycle", 16))
        runtime_registry.update(
            steps_per_cycle=self._steps_per_cycle,
            batch_size=self.device_cfg.batch_size,
            parallel_self_play_games=self._parallel_self_play_games,
            target_parallel_self_play_games=self._parallel_self_play_games,
            heuristic_bootstrap_games=int(self.raw_cfg.get("self_play", {}).get("heuristic_bootstrap_games", 80)),
        )

    def _set_parallel_self_play_games(self, count: int) -> None:
        count = max(1, min(32, int(count)))
        self._parallel_self_play_games = count
        if self._self_play_pool is not None:
            self._self_play_pool.shutdown(wait=False, cancel_futures=True)
        self._self_play_pool = ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn"))
        with self._active_games_lock:
            self._active_games = {}
        runtime_registry.update(
            parallel_self_play_games=count,
            active_games=[],
        )

    async def _run_self_play_cycle(self) -> list[SelfPlayResult]:
        if self._self_play_pool is None:
            self._set_parallel_self_play_games(self._parallel_self_play_games)
        assert self._self_play_pool is not None

        self_play_cfg = self.raw_cfg.get("self_play", {})
        total_games = max(
            self._parallel_self_play_games,
            int(self_play_cfg.get("games_per_cycle", self._parallel_self_play_games)),
        )
        tasks: list[tuple[Future[SelfPlayResult], Connection, Connection]] = []
        for game_index in range(total_games):
            worker_id = game_index % self._parallel_self_play_games
            parent_conn, child_conn = get_context("spawn").Pipe(duplex=True)
            payload = ProcessSelfPlayTask(
                worker_id=worker_id,
                played_games=self._self_play_games_completed + game_index,
                board_size=self.app_cfg.board_size,
                win_length=self.app_cfg.win_length,
                temperature=float(self.raw_cfg.get("mcts", {}).get("temperature", 1.0)),
                random_opening_moves=int(self_play_cfg.get("random_opening_moves", 2)),
                exploration_epsilon=float(self_play_cfg.get("exploration_epsilon", 0.08)),
                bootstrap_games=int(self_play_cfg.get("heuristic_bootstrap_games", 80)),
                heuristic_mix_ratio=float(self_play_cfg.get("heuristic_mix_ratio", 0.25)),
                prune_keep_ratio=float(self_play_cfg.get("heuristic_prune_keep_ratio", 0.6)),
            )
            future = self._self_play_pool.submit(play_one_game_process, payload, child_conn)
            tasks.append((future, parent_conn, child_conn))

        results: list[SelfPlayResult] = []
        pending = tasks
        while pending:
            infer_batch: list[tuple[Connection, np.ndarray, list[int]]] = []
            next_pending: list[tuple[Future[SelfPlayResult], Connection, Connection]] = []
            for future, conn, child_conn in pending:
                if future.done():
                    conn.close()
                    child_conn.close()
                    results.append(future.result())
                    continue
                if conn.poll():
                    req = conn.recv()
                    board = req.get("board")
                    legal_moves = req.get("legal_moves")
                    if isinstance(board, np.ndarray) and isinstance(legal_moves, list):
                        infer_batch.append((conn, board, legal_moves))
                next_pending.append((future, conn, child_conn))

            if infer_batch:
                infer_requests = [(board, legal_moves) for _, board, legal_moves in infer_batch]
                moves = self.trainer.infer_moves_batch(infer_requests)
                for (conn, _, _), move in zip(infer_batch, moves, strict=False):
                    conn.send(move)

            pending = next_pending
            if pending:
                await asyncio.sleep(0.001)

        self._self_play_games_completed += len(results)
        runtime_registry.update(active_games=[])
        return results

    def _on_game_progress(
        self,
        worker_id: int,
        board: object,
        move_count: int,
        winner: int,
        done: bool,
        elapsed_ms: float,
    ) -> None:
        with self._active_games_lock:
            self._active_games[worker_id] = {
                "worker_id": worker_id,
                "board": board.tolist() if hasattr(board, "tolist") else board,
                "move_count": move_count,
                "winner": winner,
                "done": done,
                "elapsed_ms": round(elapsed_ms, 1),
            }
            active = [self._active_games[i] for i in sorted(self._active_games.keys())]
        runtime_registry.update(active_games=active)

    async def run_forever(self) -> None:
        """Run forever-training cycles until interrupted."""
        cycle = 0
        best_elo = 1200.0
        # EMA 计数器
        ema_game_ms: float = 0.0
        ema_game_moves: float = 0.0
        ema_train_sps: float = 0.0
        ema_game_rate: float = 0.0
        EMA = 0.1

        while True:
            snapshot = runtime_registry.snapshot()
            target_parallel = int(snapshot.get("target_parallel_self_play_games", self._parallel_self_play_games))
            if target_parallel != self._parallel_self_play_games:
                self._set_parallel_self_play_games(target_parallel)

            if bool(snapshot.get("paused", False)):
                runtime_registry.update(status="paused")
                await asyncio.sleep(0.2)
                continue

            cycle += 1
            runtime_registry.update(status="self_play")
            t0 = time.perf_counter()
            games = await self._run_self_play_cycle()
            batch_ms = (time.perf_counter() - t0) * 1000.0
            total_moves = 0
            total_game_ms = 0.0
            total_samples = 0
            total_heuristic_moves = 0
            total_model_moves = 0
            for game in games:
                total_moves += game.moves
                total_game_ms += game.elapsed_ms
                total_samples += len(game.samples)
                total_heuristic_moves += game.heuristic_moves
                total_model_moves += game.model_moves
                for sample in game.samples:
                    self.replay.push(sample)

            game_ms = total_game_ms / max(1, len(games))
            game_moves = total_moves / max(1, len(games))
            games_per_min = (len(games) * 60_000.0) / max(batch_ms, 1.0)

            ema_game_ms = ema_game_ms * (1 - EMA) + game_ms * EMA if ema_game_ms else game_ms
            ema_game_moves = ema_game_moves * (1 - EMA) + game_moves * EMA if ema_game_moves else game_moves
            ema_game_rate = ema_game_rate * (1 - EMA) + games_per_min * EMA if ema_game_rate else games_per_min

            runtime_registry.update(
                self_play_games=runtime_registry.snapshot()["self_play_games"] + len(games),
                replay_size=len(self.replay),
                avg_game_ms=round(ema_game_ms, 1),
                avg_game_moves=int(ema_game_moves),
                games_per_min=round(ema_game_rate, 1),
                parallel_self_play_games=self._parallel_self_play_games,
                heuristic_policy_moves=int(runtime_registry.snapshot().get("heuristic_policy_moves", 0)) + total_heuristic_moves,
                model_policy_moves=int(runtime_registry.snapshot().get("model_policy_moves", 0)) + total_model_moves,
            )

            runtime_registry.update(status="training")
            train_steps_per_cycle = self._steps_per_cycle
            t1 = time.perf_counter()
            last_metrics = None
            for _ in range(train_steps_per_cycle):
                last_metrics = self.trainer.train_step()
            train_elapsed = time.perf_counter() - t1
            sps = train_steps_per_cycle / max(train_elapsed, 1e-6)
            ema_train_sps = ema_train_sps * (1 - EMA) + sps * EMA if ema_train_sps else sps
            if last_metrics is not None:
                metrics = last_metrics
            runtime_registry.update(
                training_step=metrics.step,
                latest_loss=metrics.loss,
                train_steps_per_sec=round(ema_train_sps, 1),
            )
            switched_model = training_bridge.maybe_switch_model(metrics.step)
            if switched_model is not None:
                runtime_registry.update(current_model=switched_model)

            if cycle % 20 == 0:
                runtime_registry.update(status="arena")
                result = self.arena.evaluate(self.trainer.infer_move, best_elo)
                best_elo = result.best_elo if not result.promoted else result.challenger_elo
                runtime_registry.update(
                    arena_games=runtime_registry.snapshot()["arena_games"] + result.games,
                    latest_elo=best_elo,
                    best_model="candidate" if result.promoted else "bootstrap",
                )

            runtime_registry.update(status="idle")
            await asyncio.sleep(0.05)
