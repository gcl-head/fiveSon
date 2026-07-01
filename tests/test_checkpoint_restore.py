from pathlib import Path

import numpy as np
import pytest

from replay_buffer.prioritized import PrioritizedReplayBuffer, ReplaySample
from training.trainer import Trainer


def test_trainer_checkpoint_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    replay = PrioritizedReplayBuffer(capacity=8, prioritized=True, alpha=0.6)
    board = np.zeros((15, 15), dtype=np.float32)
    policy = np.zeros(225, dtype=np.float32)
    policy[0] = 1.0
    replay.push(ReplaySample(state=board, policy_target=policy, value_target=1.0, priority=1.0))

    trainer = Trainer(replay, board_size=15, batch_size=1, device="cpu", amp_enabled=False, learning_rate=0.001)
    metrics = trainer.train_step()
    assert metrics.step == 1

    checkpoint_dir = tmp_path / "checkpoints"
    saved = trainer.save_checkpoint(
        checkpoint_dir,
        generation=3,
        current_model="checkpoint-g3-s1",
        best_model="candidate",
    )
    assert saved is not None

    restored = Trainer(
        PrioritizedReplayBuffer(capacity=8, prioritized=True, alpha=0.6),
        board_size=15,
        batch_size=1,
        device="cpu",
        amp_enabled=False,
        learning_rate=0.001,
    )
    meta = restored.load_checkpoint(checkpoint_dir, board_size=15, action_dim=225)
    assert meta is not None
    assert meta["training_step"] == 1
    assert meta["generation"] == 3
    assert meta["current_model"] == "checkpoint-g3-s1"
    assert meta["best_model"] == "candidate"

    move = restored.infer_move(board, [0, 1, 2])
    assert move is not None