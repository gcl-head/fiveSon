# Architecture Notes

## Long-running Loop

1. Self-play workers generate trajectories from current model policy.
2. Trajectories are transformed into policy/value targets and pushed to replay buffer.
3. Trainer samples replay batches and updates model parameters.
4. Arena runs challenger vs best snapshots and computes promotion decision.
5. Best model pointer is updated when win-rate threshold is reached.

## Separation of Concerns

- `backend/`: serving and orchestration visibility
- `engine/`: deterministic game rules
- `network/`: replaceable neural backbones (ResNet now, Transformer later)
- `mcts/`: search policy implementation (PUCT now, Gumbel/MuZero extension path)
- `training/`: forever loop, optimizer/scheduler/checkpoint control
- `arena/`: model gating and ELO

## Device Policy

- Prefer `mps` when `torch.backends.mps.is_available()` and `is_built()` are both true.
- Fallback to `cuda` if available.
- Fallback to CPU only when no accelerator exists.
- Pin memory defaults to false on MPS.

## Data Layout

- Metadata and run state in SQLite.
- Large sample payloads and aggregated analytics in Parquet.
- Structured logs in CSV/JSON and TensorBoard scalars.
