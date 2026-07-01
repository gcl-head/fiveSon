"""Replay buffer with FIFO + optional prioritized sampling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ReplaySample:
    """Single replay training sample."""

    state: Any
    policy_target: Any
    value_target: float
    priority: float = 1.0


class PrioritizedReplayBuffer:
    """Bounded replay buffer with proportional priority sampling."""

    def __init__(self, capacity: int, prioritized: bool = True, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self._items: deque[ReplaySample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def push(self, sample: ReplaySample) -> None:
        self._items.append(sample)

    def sample(self, batch_size: int) -> list[ReplaySample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._items) == 0:
            return []

        batch_size = min(batch_size, len(self._items))
        items = list(self._items)

        if not self.prioritized:
            idx = np.random.choice(len(items), size=batch_size, replace=False)
            return [items[int(i)] for i in idx]

        priorities = np.array([max(1e-6, s.priority) for s in items], dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        idx = np.random.choice(len(items), size=batch_size, replace=False, p=probs)
        return [items[int(i)] for i in idx]
