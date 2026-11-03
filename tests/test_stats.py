import numpy as np
import pytest

from finqa_ft.stats import ci, mcnemar_exact, paired_diff, resample_idx


def test_resample_is_seeded_and_shared():
    a, b = resample_idx(50, 100, seed=1), resample_idx(50, 100, seed=1)
    assert a.shape == (100, 50) and (a == b).all() and a.max() < 50


def test_ci_contains_mean_and_shrinks_with_n():
    x = np.array([1] * 70 + [0] * 30)
    mean, lo, hi = ci(x, resample_idx(len(x)))
    assert mean == 0.7 and lo < 0.7 < hi
    big = np.tile(x, 10)
    _, lo2, hi2 = ci(big, resample_idx(len(big)))
    assert hi2 - lo2 < hi - lo
    assert ci(np.ones(10), resample_idx(10)) == (1.0, 1.0, 1.0)


def test_paired_diff_identical_models_is_zero():
    x = np.array([1, 0, 1, 1, 0] * 20)
    assert paired_diff(x, x, resample_idx(len(x))) == (0.0, 0.0, 0.0)


def test_mcnemar_exact():
    a = np.array([1, 1, 1, 1, 1, 0, 1, 0])
    b = np.array([0, 0, 0, 0, 0, 0, 1, 0])
    r = mcnemar_exact(a, b)
    assert (r["only_a"], r["only_b"]) == (5, 0)
    assert r["p"] == pytest.approx(2 / 32)
    assert mcnemar_exact(a, a)["p"] == 1.0
    # 10 vs 3 discordant: two-sided binomial p = 0.0923...
    a = np.array([1] * 10 + [0] * 3)
    assert mcnemar_exact(a, 1 - a)["p"] == pytest.approx(0.092285, abs=1e-6)
