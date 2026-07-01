"""ELO update helpers for arena promotions."""

from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score of A against B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating: float, expected: float, actual: float, k: float = 32.0) -> float:
    """Apply one-step ELO update."""
    return rating + k * (actual - expected)
