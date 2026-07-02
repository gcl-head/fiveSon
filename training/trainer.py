"""Training loop scaffolding with perpetual cycle support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from replay_buffer.prioritized import PrioritizedReplayBuffer


@dataclass(slots=True)
class TrainMetrics:
    """Latest training metrics emitted to runtime dashboard."""

    step: int
    loss: float


class Trainer:
    """Trainer with torch fallback behavior when GPU stack is unavailable."""

    def __init__(
        self,
        replay: PrioritizedReplayBuffer,
        board_size: int,
        batch_size: int,
        device: str,
        amp_enabled: bool,
        learning_rate: float,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.replay = replay
        self.board_size = int(board_size)
        self.batch_size = batch_size
        self.device = device
        self.amp_enabled = amp_enabled and device in {"cuda", "mps"}
        self.learning_rate = learning_rate
        self.step = 0
        self.checkpoint_dir = checkpoint_dir

        self._torch: Any | None = None
        self._model: Any | None = None
        self._optimizer: Any | None = None
        self._lock = Lock()

        try:
            import torch

            self._torch = torch
        except Exception:
            self._torch = None

        if self._torch is not None:
            self._ensure_model(self.board_size, self.board_size * self.board_size)

    def _ensure_model(self, board_size: int, action_dim: int) -> bool:
        if self._torch is None:
            return False
        if self._model is not None:
            return True

        torch = self._torch
        nn = torch.nn
        F = torch.nn.functional

        num_res_blocks = 20
        num_channels = 256

        class SEBlock(nn.Module):  # type: ignore[name-defined]
            def __init__(self, channels: int, reduction: int = 16) -> None:
                super().__init__()
                hidden = max(1, channels // reduction)
                self.fc1 = nn.Linear(channels, hidden)
                self.fc2 = nn.Linear(hidden, channels)

            def forward(self, x: Any) -> Any:
                b, c, _, _ = x.shape
                y = x.mean(dim=(2, 3))
                y = F.relu(self.fc1(y))
                y = torch.sigmoid(self.fc2(y))
                y = y.view(b, c, 1, 1)
                return x * y

        class ResidualBlock(torch.nn.Module):  # type: ignore[name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(num_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(num_channels),
                )
                self.se = SEBlock(num_channels)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Any) -> Any:
                out = self.net(x)
                out = self.se(out)
                return self.relu(out + x)

        class PolicyValueNet(torch.nn.Module):  # type: ignore[name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(1, num_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(num_channels),
                    nn.ReLU(inplace=True),
                )
                self.blocks = nn.Sequential(*[ResidualBlock() for _ in range(num_res_blocks)])
                self.policy_head = nn.Sequential(
                    nn.Conv2d(num_channels, 64, 1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, 1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, 1, bias=True),
                )
                self.value_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(num_channels, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 1),
                    nn.Tanh(),
                )

            def forward(self, x: Any) -> tuple[Any, Any]:
                h = self.blocks(self.stem(x))
                policy = self.policy_head(h)
                policy = policy.view(policy.shape[0], -1)
                return policy, self.value_head(h)

        self._model = PolicyValueNet().to(self.device)
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return True

    def _checkpoint_paths(self, checkpoint_dir: Path, generation: int) -> tuple[Path, Path]:
        generation = max(0, int(generation))
        return checkpoint_dir / f"generation-g{generation}.pt", checkpoint_dir / f"generation-g{generation}.json"

    def _write_checkpoint_payload(
        self,
        checkpoint_dir: Path,
        generation: int,
        current_model: str,
        best_model: str,
    ) -> Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "latest.pt"
        payload = {
            "training_step": int(self.step),
            "generation": max(0, int(generation)),
            "current_model": current_model,
            "best_model": best_model,
            "device": self.device,
            "amp_enabled": bool(self.amp_enabled),
            "learning_rate": float(self.learning_rate),
        }
        if self._torch is not None and self._model is not None and self._optimizer is not None:
            payload["model_state_dict"] = self._model.state_dict()
            payload["optimizer_state_dict"] = self._optimizer.state_dict()
            self._torch.save(payload, checkpoint_path)
        else:
            checkpoint_path = checkpoint_dir / "latest.json"
            checkpoint_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        metadata_path = checkpoint_dir / "latest.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "training_step": int(self.step),
                    "generation": max(0, int(generation)),
                    "current_model": current_model,
                    "best_model": best_model,
                    "device": self.device,
                    "amp_enabled": bool(self.amp_enabled),
                    "learning_rate": float(self.learning_rate),
                    "checkpoint_path": "latest.pt" if checkpoint_path.suffix == ".pt" else checkpoint_path.name,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return checkpoint_path

    def _load_checkpoint_payload(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str,
        metadata_name: str,
        board_size: int,
        action_dim: int,
    ) -> dict[str, Any] | None:
        metadata_path = checkpoint_dir / metadata_name
        checkpoint_path = checkpoint_dir / checkpoint_name
        if not metadata_path.exists() and not checkpoint_path.exists():
            return None

        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}

        with self._lock:
            if self._torch is not None and checkpoint_path.exists() and checkpoint_path.suffix == ".pt":
                checkpoint = self._torch.load(checkpoint_path, map_location=self.device)
                self._ensure_model(board_size=board_size, action_dim=action_dim)
                assert self._model is not None
                assert self._optimizer is not None
                try:
                    self._model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer_state = checkpoint.get("optimizer_state_dict")
                    if optimizer_state is not None:
                        self._optimizer.load_state_dict(optimizer_state)
                except Exception:
                    self._model = None
                    self._optimizer = None
                    self.step = 0
                    self._ensure_model(board_size=board_size, action_dim=action_dim)
                    return None
                self.step = int(checkpoint.get("training_step", metadata.get("training_step", 0)))
                return {
                    "training_step": self.step,
                    "generation": int(checkpoint.get("generation", metadata.get("generation", 0))),
                    "current_model": str(checkpoint.get("current_model", metadata.get("current_model", "bootstrap"))),
                    "best_model": str(checkpoint.get("best_model", metadata.get("best_model", "bootstrap"))),
                }

            self.step = int(metadata.get("training_step", 0))
            return {
                "training_step": self.step,
                "generation": int(metadata.get("generation", 0)),
                "current_model": str(metadata.get("current_model", "bootstrap")),
                "best_model": str(metadata.get("best_model", "bootstrap")),
            }

    def train_step(self) -> TrainMetrics:
        with self._lock:
            batch = self.replay.sample(self.batch_size)
            self.step += 1

            if not batch:
                return TrainMetrics(step=self.step, loss=0.0)

            state_np = np.stack([np.asarray(sample.state, dtype=np.float32) for sample in batch], axis=0)
            policy_np = np.stack([np.asarray(sample.policy_target, dtype=np.float32) for sample in batch], axis=0)
            value_np = np.asarray([float(sample.value_target) for sample in batch], dtype=np.float32)

            board_size = int(state_np.shape[-1])
            action_dim = int(policy_np.shape[-1])

            if not self._ensure_model(board_size=board_size, action_dim=action_dim):
                # Torch unavailable fallback.
                loss = 1.0 / (1.0 + len(batch))
                return TrainMetrics(step=self.step, loss=loss)

            torch = self._torch
            assert torch is not None
            assert self._model is not None
            assert self._optimizer is not None

            x = torch.from_numpy(state_np).reshape(len(batch), 1, board_size, board_size).to(self.device)
            policy_target = torch.from_numpy(policy_np).to(self.device)
            value_target = torch.from_numpy(value_np).to(self.device)

            policy_logits, value_pred = self._model(x)

            policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
            policy_loss = -(policy_target * policy_log_probs).sum(dim=-1).mean()
            value_loss = torch.mean((value_pred.squeeze(-1) - value_target) ** 2)
            total_loss = policy_loss + value_loss

            self._optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self._optimizer.step()

            return TrainMetrics(step=self.step, loss=float(total_loss.detach().item()))

    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        generation: int,
        current_model: str,
        best_model: str,
    ) -> Path | None:
        """Persist the latest trainable state so restarts can resume from disk."""
        with self._lock:
            latest_path = self._write_checkpoint_payload(checkpoint_dir, generation, current_model, best_model)

            generation_path, generation_metadata_path = self._checkpoint_paths(checkpoint_dir, generation)
            if latest_path.suffix == ".pt" and self._torch is not None and self._model is not None and self._optimizer is not None:
                generation_payload = {
                    "training_step": int(self.step),
                    "generation": max(0, int(generation)),
                    "current_model": current_model,
                    "best_model": best_model,
                    "device": self.device,
                    "amp_enabled": bool(self.amp_enabled),
                    "learning_rate": float(self.learning_rate),
                    "model_state_dict": self._model.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                }
                self._torch.save(generation_payload, generation_path)
                generation_metadata_path.write_text(
                    json.dumps(
                        {
                            "training_step": int(self.step),
                            "generation": max(0, int(generation)),
                            "current_model": current_model,
                            "best_model": best_model,
                            "device": self.device,
                            "amp_enabled": bool(self.amp_enabled),
                            "learning_rate": float(self.learning_rate),
                            "checkpoint_path": generation_path.name,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            return latest_path

    def load_checkpoint(self, checkpoint_dir: Path, board_size: int, action_dim: int) -> dict[str, Any] | None:
        """Restore the latest trainable state if a checkpoint exists."""
        return self._load_checkpoint_payload(checkpoint_dir, "latest.pt", "latest.json", board_size, action_dim)

    def load_generation_checkpoint(
        self,
        checkpoint_dir: Path,
        generation: int,
        board_size: int,
        action_dim: int,
    ) -> dict[str, Any] | None:
        """Restore a generation-specific checkpoint if it exists."""
        generation_path, generation_metadata_path = self._checkpoint_paths(checkpoint_dir, generation)
        return self._load_checkpoint_payload(
            checkpoint_dir,
            generation_path.name,
            generation_metadata_path.name,
            board_size,
            action_dim,
        )

    def infer_move(self, board: np.ndarray, legal_moves: list[int]) -> int | None:
        """Infer one move from the latest trained model; returns None when unavailable."""
        moves = self.infer_moves_batch([(board, legal_moves)])
        return moves[0] if moves else None

    def infer_moves_batch(self, requests: list[tuple[np.ndarray, list[int]]]) -> list[int | None]:
        """Infer legal best moves for a batch of boards in one forward pass."""
        if not requests:
            return []

        with self._lock:
            if self._torch is None or self._model is None or self.step <= 0:
                return [None for _ in requests]

            torch = self._torch
            assert torch is not None

            boards = [np.asarray(board, dtype=np.float32) for board, _ in requests]
            board_size = int(boards[0].shape[-1])
            batch = np.stack(boards, axis=0)
            x = torch.from_numpy(batch).reshape(len(boards), 1, board_size, board_size).to(self.device)

            self._model.eval()
            with torch.no_grad():
                policy_logits, _ = self._model(x)
            self._model.train()

            logits_np = policy_logits.detach().cpu().numpy()
            moves: list[int | None] = []
            action_dim = board_size * board_size
            for i, (_, legal_moves) in enumerate(requests):
                if not legal_moves:
                    moves.append(None)
                    continue
                legal_idx = np.asarray(legal_moves, dtype=np.int64)
                best_legal = int(legal_idx[np.argmax(logits_np[i, legal_idx])])
                moves.append(best_legal if best_legal < action_dim else None)
            return moves
