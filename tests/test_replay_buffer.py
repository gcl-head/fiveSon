from replay_buffer.prioritized import PrioritizedReplayBuffer, ReplaySample


def test_replay_push_and_sample() -> None:
    replay = PrioritizedReplayBuffer(capacity=8, prioritized=True, alpha=0.6)
    for idx in range(6):
        replay.push(ReplaySample(state=idx, policy_target=idx, value_target=1.0, priority=idx + 1))

    assert len(replay) == 6
    batch = replay.sample(batch_size=4)
    assert len(batch) == 4
