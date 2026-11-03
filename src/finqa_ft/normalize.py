import math

REL_TOL = 1e-3
ABS_TOL = 1e-5


def to_answer(value: float | str | None) -> float | str | None:
    """Normalise an executed or gold answer to a float or 'yes'/'no'."""
    if value is None or isinstance(value, float | int):
        return None if value is None or not math.isfinite(value) else float(value)
    t = value.strip().lower()
    if t in ("yes", "no", "true", "false"):
        return "yes" if t in ("yes", "true") else "no"
    try:
        return float(t.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def answers_match(pred: float | str | None, gold: float | str | None, strict: bool = False) -> bool:
    """Execution-accuracy match.

    Numbers match within 0.1% relative tolerance. Unless strict, a prediction that is off
    by exactly a factor of 100 also matches (percent vs fraction, e.g. 0.12 vs 12%).
    """
    p, g = to_answer(pred), to_answer(gold)
    if p is None or g is None:
        return False
    if isinstance(p, str) or isinstance(g, str):
        return p == g
    candidates = [p] if strict else [p, p * 100, p / 100]
    return any(math.isclose(c, g, rel_tol=REL_TOL, abs_tol=ABS_TOL) for c in candidates)
