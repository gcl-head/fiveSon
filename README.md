# fiveSon: Gomoku AI Platform (KataGo + AlphaZero Style)

`fiveSon` is an engineering-first Gomoku AI platform for long-running self-play, training, evaluation, and model promotion.

## Current Scope

- System probe and hardware-aware configuration generation
- FastAPI backend with websocket status stream
- Modular architecture for network, MCTS, self-play, replay buffer, training, and arena
- YAML-first configuration with no hard-coded training constants
- SQLite/Parquet-ready data paths
- Test, lint, and type-check toolchain

## Quick Start

1. Create and activate a Python environment (recommended Python 3.11 or 3.12).
2. Install dependencies:

```bash
pip install -e .[dev]
# For training stack (requires Python < 3.13 currently for torch wheels)
pip install -e .[train]
```

3. Generate system report:

```bash
python tools/system_probe.py --output system_report.md
```

4. Start backend:

```bash
python scripts/run_backend.py
```

Recommended for production-like long runs (auto health-check and restart):

```bash
python scripts/backend_guard.py
```

Behavior defaults for long-running stability:

- Overload guard will back off self-play parallelism first and will not auto-pause by default.
- Watchdog monitors both `/api/health` and `/api/status` progress.
- If runtime is auto-paused for too long or training progress stalls, watchdog triggers recovery (resume or restart).

Then open `http://127.0.0.1:8000` to access the control center.

- The orchestrator loop auto-starts with the backend process.
- The dashboard updates runtime metrics by websocket.
- The board supports click-to-play human vs AI with New Game reset.
- When pressure spikes, overload guard will reduce parallel self-play automatically and can auto-pause training.

5. (Optional) Start standalone orchestrator loop (self-play/train/eval skeleton):

```bash
python scripts/run_orchestrator.py
```

## Project Layout

- `backend/`: FastAPI app, websocket, runtime state
- `engine/`: board logic and game rules
- `network/`: neural network interfaces and torch implementation
- `mcts/`: PUCT MCTS and tree policies
- `self_play/`: data generation workers
- `replay_buffer/`: prioritized replay storage and sampling
- `training/`: forever training loop and checkpoints
- `arena/`: challenger-vs-best evaluation and promotion policy
- `evaluation/`: metrics and reports
- `configs/`: YAML configs and tuning profiles
- `database/`: SQLite data storage
- `logs/`: structured runtime logs
- `checkpoints/`: periodic snapshots
- `weights/`: promoted model files
- `docs/`: architecture and operations docs

## Notes on Apple Silicon

- The runtime prefers `mps` when available and built in torch.
- Pin memory is disabled by default on MPS and enabled on CUDA/CPU as appropriate.
- If torch is not available in the active Python environment, the system can run service mode and dry-run orchestration but training is disabled until torch is installed.

## Development

```bash
ruff check .
mypy .
pytest
```

## Roadmap

- Residual network + policy/value heads fully wired to self-play targets
- Batched GPU inference server for MCTS workers
- Arena ELO with SPRT gating
- Curriculum progression and opening/endgame databases
- Distributed self-play and training
