"""Training loop scaffolding with perpetual cycle support."""

from __future__ import annotations

from dataclasses import dataclass
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
        batch_size: int,
        device: str,
        amp_enabled: bool,
        learning_rate: float,
    ) -> None:
        self.replay = replay
        self.batch_size = batch_size
        self.device = device
        self.amp_enabled = amp_enabled and device in {"cuda", "mps"}
        self.learning_rate = learning_rate
        self.step = 0

        self._torch: Any | None = None
        self._model: Any | None = None
        self._optimizer: Any | None = None
        self._lock = Lock()

        try:
            import torch

            self._torch = torch
        except Exception:
            self._torch = None

    def _ensure_model(self, board_size: int, action_dim: int) -> bool:
        if self._torch is None:
            return False
        if self._model is not None:
            return True

        torch = self._torch
        nn = torch.nn

        channels = 128
        num_blocks = 6

        class ResBlock(torch.nn.Module):  # type: ignore[name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Any) -> Any:
                return self.relu(self.net(x) + x)

        class PolicyValueNet(torch.nn.Module):  # type: ignore[name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(1, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                self.blocks = nn.Sequential(*[ResBlock() for _ in range(num_blocks)])
                # policy head
                self.p_conv = nn.Sequential(
                    nn.Conv2d(channels, 2, 1, bias=False),
                    nn.BatchNorm2d(2),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                )
                self.p_fc = nn.Linear(2 * board_size * board_size, action_dim)
                # value head
                self.v_conv = nn.Sequential(
                    nn.Conv2d(channels, 1, 1, bias=False),
                    nn.BatchNorm2d(1),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                )
                self.v_fc = nn.Sequential(
                    nn.Linear(board_size * board_size, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 1),
                    nn.Tanh(),
                )

            def forward(self, x: Any) -> tuple[Any, Any]:
                h = self.blocks(self.stem(x))
                return self.p_fc(self.p_conv(h)), self.v_fc(self.v_conv(h))

        self._model = PolicyValueNet().to(self.device)
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return True

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

    def infer_move(self, board: np.ndarray, legal_moves: list[int]) -> int | None:
        """Infer one move from the latest trained model; returns None when unavailable."""
        if not legal_moves:
            return None

        with self._lock:
            if self._torch is None or self._model is None or self.step <= 0:
                return None

            torch = self._torch
            assert torch is not None
            board_size = int(board.shape[-1])
            action_dim = board_size * board_size
            x = torch.from_numpy(board.astype(np.float32)).reshape(1, 1, board_size, board_size).to(self.device)

            self._model.eval()
            with torch.no_grad():
                policy_logits, _ = self._model(x)
            self._model.train()

            logits = policy_logits[0].detach().cpu().numpy()
            legal_idx = np.asarray(legal_moves, dtype=np.int64)
            best_legal = int(legal_idx[np.argmax(logits[legal_idx])])
            if best_legal >= action_dim:
                return None
            return best_legal
