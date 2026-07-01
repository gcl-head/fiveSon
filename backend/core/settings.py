"""Configuration loading and hardware-aware runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(slots=True)
class DeviceConfig:
    """Device and dataloader settings selected at runtime."""

    device: str
    amp_enabled: bool
    torch_compile_enabled: bool
    gradient_checkpointing: bool
    batch_size: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool


@dataclass(slots=True)
class AppConfig:
    """Top-level application config parsed from YAML."""

    project_name: str
    board_size: int
    win_length: int
    host: str
    port: int
    db_url: str
    replay_capacity: int
    arena_games: int
    promotion_win_rate: float


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def load_app_config(path: Path) -> AppConfig:
    """Parse the app config section from YAML."""
    raw = load_yaml(path)
    app = raw["app"]
    replay = raw["replay_buffer"]
    arena = raw["arena"]
    return AppConfig(
        project_name=str(app["project_name"]),
        board_size=int(app["board_size"]),
        win_length=int(app["win_length"]),
        host=str(app["host"]),
        port=int(app["port"]),
        db_url=str(app["database_url"]),
        replay_capacity=int(replay["capacity"]),
        arena_games=int(arena["games"]),
        promotion_win_rate=float(arena["promotion_win_rate"]),
    )


def choose_device_config(path: Path) -> DeviceConfig:
    """Select best-effort runtime knobs based on YAML + detected hardware."""
    raw = load_yaml(path)
    train = raw["training"]

    try:
        import torch

        mps_available = bool(torch.backends.mps.is_available())
        mps_built = bool(torch.backends.mps.is_built())
        cuda_available = bool(torch.cuda.is_available())
        compile_ok = hasattr(torch, "compile")
    except Exception:
        mps_available = False
        mps_built = False
        cuda_available = False
        compile_ok = False

    if mps_available and mps_built:
        device = "mps"
        pin_memory = False
    elif cuda_available:
        device = "cuda"
        pin_memory = True
    else:
        device = "cpu"
        pin_memory = False

    cpu_workers = int(train.get("max_num_workers", 4))
    prefetch = int(train.get("prefetch_factor", 2))

    return DeviceConfig(
        device=device,
        amp_enabled=bool(train.get("amp", True)) and device in {"cuda", "mps"},
        torch_compile_enabled=bool(train.get("torch_compile", True)) and compile_ok,
        gradient_checkpointing=bool(train.get("gradient_checkpointing", True)),
        batch_size=int(train.get("batch_size", 256)),
        num_workers=max(1, cpu_workers),
        prefetch_factor=max(1, prefetch),
        pin_memory=pin_memory,
    )
