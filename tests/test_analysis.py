import pytest
from conftest import make

from finqa_ft.analysis import decompose, unseen_mask, wilson, x100_flips
from finqa_ft.metrics import score


def test_unseen_mask_uses_company_year():
    exs = {
        e.id: e for e in (make("A/2010/p.pdf-1"), make("A/2011/p.pdf-1"), make("B/2010/p.pdf-2"))
    }
    mask = unseen_mask(list(exs), exs, {"A/2010", "B/2009"})
    assert mask.tolist() == [False, True, True]


def test_x100_flips_counts_only_lenient_rescues():
    ex = make(program="divide(1, 4)", answer=0.25)
    rows = [score(ex, "Program: divide(1, 4)"), score(ex, "Program: divide(100, 4)")]
    rows.append(score(ex, "Program: divide(3, 4)"))
    assert [(r["correct"], r["correct_strict"]) for r in rows] == [
        (True, True),
        (True, False),
        (False, False),
    ]
    assert x100_flips(rows) == 1
    assert not rows[1]["official_exec"]


def test_decompose_parts_add_up():
    d = decompose({"valid": 0.9, "acc_valid": 0.6}, {"valid": 1.0, "acc_valid": 0.75})
    assert d["total"] == pytest.approx(0.75 - 0.54)
    assert d["format"] + d["correctness"] == pytest.approx(d["total"])
    assert d["format"] == pytest.approx(0.1 * 0.675)
    no_format = decompose({"valid": 1.0, "acc_valid": 0.5}, {"valid": 1.0, "acc_valid": 0.6})
    assert no_format["format"] == 0


def test_wilson():
    lo, hi = wilson(37, 100)
    assert lo == pytest.approx(0.282, abs=1e-3) and hi == pytest.approx(0.468, abs=1e-3)
