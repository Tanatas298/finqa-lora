"""Pure helpers for the results analysis (used by scripts/make_tables.py)."""

import numpy as np


def unseen_mask(test_ids: list[str], exs: dict, train_reports: set[str]) -> np.ndarray:
    """True where the question's company-year (e.g. 'ETR/2016') never occurs in train."""
    return np.array([exs[i].report not in train_reports for i in test_ids])


def x100_flips(rows: list[dict]) -> int:
    """Predictions that are wrong without the x100 allowance and right with it."""
    return sum(r["correct"] and not r["correct_strict"] for r in rows)


def decompose(base: dict, new: dict) -> dict:
    """Split an accuracy gain into format (valid rate) and correctness among valid programs.

    acc = valid * acc_valid. The two one-factor-first orderings are averaged (Shapley split),
    so the two parts add up exactly to the total gain.
    """
    v0, a0, v1, a1 = base["valid"], base["acc_valid"], new["valid"], new["acc_valid"]
    fmt = ((v1 - v0) * a0 + (v1 - v0) * a1) / 2
    corr = ((a1 - a0) * v0 + (a1 - a0) * v1) / 2
    return {"total": v1 * a1 - v0 * a0, "format": fmt, "correctness": corr}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)
