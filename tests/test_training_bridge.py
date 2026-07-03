import numpy as np

from backend.services.training_bridge import TrainingBridge
from replay_buffer.prioritized import PrioritizedReplayBuffer


def test_model_switch_interval() -> None:
    bridge = TrainingBridge()
    bridge.configure({"webplay": {"model_switch_interval_steps": 10}})

    assert bridge.candidate_generation(0) == 0
    assert bridge.candidate_generation(9) == 0
    assert bridge.candidate_generation(10) == 1

    assert bridge.maybe_switch_model(9) is None
    first = bridge.maybe_switch_model(10)
    assert first is not None
    assert "g1" in first

    assert bridge.maybe_switch_model(19) is None
    second = bridge.maybe_switch_model(20)
    assert second is not None
    assert "g2" in second


def test_promote_generation_requires_candidate_progress() -> None:
    bridge = TrainingBridge()
    bridge.configure({"webplay": {"model_switch_interval_steps": 10}})

    assert bridge.promote_generation_if_eligible(9) is None
    promoted = bridge.promote_generation_if_eligible(10)
    assert promoted is not None
    generation, model_name = promoted
    assert generation == 1
    assert "g1" in model_name

    # No duplicate promotion for the same candidate generation.
    assert bridge.promote_generation_if_eligible(19) is None


def test_human_ingestion_uses_high_priority() -> None:
    bridge = TrainingBridge()
    replay = PrioritizedReplayBuffer(capacity=32, prioritized=True, alpha=0.6)
    bridge.configure({"webplay": {"human_game_priority_weight": 12.0}})
    bridge.attach_replay_buffer(replay)

    board = np.zeros((15, 15), dtype=np.int8)
    trajectory = [(board, 0, 1), (board, 1, -1), (board, 2, 1)]
    inserted = bridge.ingest_human_game(trajectory=trajectory, winner=1, board_size=15)

    assert inserted == 3
    assert len(replay) == 3
    sampled = replay.sample(batch_size=3)
    assert all(item.priority == 12.0 for item in sampled)