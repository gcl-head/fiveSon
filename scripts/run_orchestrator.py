"""Run forever orchestrator loop."""

from __future__ import annotations

import asyncio
from pathlib import Path

from training.orchestrator import Orchestrator


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    orch = Orchestrator(project_root / "configs/default.yaml")
    asyncio.run(orch.run_forever())


if __name__ == "__main__":
    main()
