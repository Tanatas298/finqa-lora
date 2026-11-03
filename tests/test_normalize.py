import pytest

from finqa_ft.normalize import answers_match, to_answer


@pytest.mark.parametrize(
    "pred,gold,ok",
    [
        (0.01639, 0.01639, True),
        (0.0163912, 0.01639, True),
        (1.639, 0.01639, True),
        (0.0001639, 0.01639, True),
        (0.0165, 0.01639, False),
        (94.0, 94.0, True),
        (94.05, 94.0, True),
        (95.0, 94.0, False),
        (-94.0, 94.0, False),
        (0.0, 0.0, True),
        (1e-6, 0.0, True),
        ("yes", "yes", True),
        ("Yes", "yes", True),
        ("no", "yes", False),
        (1.0, "yes", False),
        (None, 1.0, False),
        (float("nan"), 1.0, False),
        ("12%", 12.0, True),
    ],
)
def test_answers_match(pred, gold, ok):
    assert answers_match(pred, gold) is ok


def test_strict_rejects_percent_scale():
    assert not answers_match(1.639, 0.01639, strict=True)
    assert answers_match(0.016391, 0.01639, strict=True)


def test_to_answer():
    assert to_answer("1,234") == 1234.0
    assert to_answer("true") == "yes"
    assert to_answer("abc") is None
