"""Neural model interfaces and a residual policy-value network implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class NetworkOutput:
    """Batched inference output."""

    policy_logits: object
    value: object


class PolicyValueNetwork(Protocol):
    """Minimal interface for policy-value backbones."""

    def infer(self, features: object) -> NetworkOutput:
        """Run inference on features and return policy/value outputs."""


class TorchPolicyValueNetwork:
    """Deferred torch network wrapper to keep boot mode import-safe."""

    def __init__(self, board_size: int, channels: int = 128, blocks: int = 8) -> None:
        self.board_size = board_size
        self.channels = channels
        self.blocks = blocks
        self._model: Any | None = None

    def build(self) -> None:
        """Build torch model lazily if torch is installed."""
        import torch.nn as nn

        class ResidualBlock(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(width, width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                )
                self.act = nn.ReLU(inplace=True)

            def forward(self, x: Any) -> Any:
                return self.act(x + self.block(x))

        class Net(nn.Module):
            def __init__(self, board_size: int, channels: int, blocks: int) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
                self.policy = nn.Sequential(
                    nn.Conv2d(channels, 2, 1),
                    nn.BatchNorm2d(2),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                    nn.Linear(2 * board_size * board_size, board_size * board_size),
                )
                self.value = nn.Sequential(
                    nn.Conv2d(channels, 1, 1),
                    nn.BatchNorm2d(1),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                    nn.Linear(board_size * board_size, channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels, 1),
                    nn.Tanh(),
                )

            def forward(self, x: Any) -> tuple[Any, Any]:
                h = self.body(self.stem(x))
                return self.policy(h), self.value(h)

        self._model = Net(self.board_size, self.channels, self.blocks)

    def infer(self, features: object) -> NetworkOutput:
        if self._model is None:
            self.build()
        assert self._model is not None

        policy_logits, value = self._model(features)
        return NetworkOutput(policy_logits=policy_logits, value=value)
