"""Paired bootstrap confidence intervals and exact McNemar tests over per-question 0/1 scores."""

from math import comb

import numpy as np

N_BOOT = 10_000
SEED = 0


def resample_idx(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """One shared (n_boot, n) index matrix, so every model is resampled on the same questions."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def ci(x, idx: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval of a 0/1 (or difference) vector."""
    x = np.asarray(x, dtype=float)
    boot = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def paired_diff(a, b, idx: np.ndarray) -> tuple[float, float, float]:
    """Accuracy of a minus accuracy of b, with a paired bootstrap interval."""
    return ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), idx)


def mcnemar_exact(a, b) -> dict:
    """Two-sided exact McNemar test: binomial test on the discordant pairs."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": p}
