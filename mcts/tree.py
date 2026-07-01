"""PUCT MCTS skeleton with root noise and virtual loss hooks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class Node:
    """Single MCTS node storing visit and value statistics."""

    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, Node] = field(default_factory=dict)

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSSearch:
    """Search engine implementing core PUCT selection behavior."""

    def __init__(self, c_puct: float, virtual_loss: float) -> None:
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss

    def select_child(self, parent: Node) -> tuple[int, Node]:
        """Select action/node by PUCT score."""
        best_action = -1
        best_node = None
        best_score = float("-inf")
        sqrt_visits = math.sqrt(max(1, parent.visit_count))

        for action, child in parent.children.items():
            u = self.c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
            score = child.q_value() + u
            if score > best_score:
                best_score = score
                best_action = action
                best_node = child

        if best_node is None:
            raise RuntimeError("no children available")

        return best_action, best_node

    def add_root_dirichlet_noise(
        self,
        priors: dict[int, float],
        alpha: float,
        epsilon: float,
    ) -> dict[int, float]:
        """Blend priors with Dirichlet noise for exploration."""
        import numpy as np

        actions = list(priors)
        noise = np.random.dirichlet([alpha] * len(actions))
        mixed: dict[int, float] = {}
        for idx, action in enumerate(actions):
            mixed[action] = (1 - epsilon) * priors[action] + epsilon * float(noise[idx])
        return mixed
