import math

from conftest import make

from finqa_ft.metrics import score, summarize


def test_score_and_summarize():
    a = make("A/1/p.pdf-1", program="subtract(5, 3)", answer=2.0)
    b = make(
        "B/1/p.pdf-1", program="add(1, 2), divide(#0, 3)", answer=1.0, gold_inds={"text_0": "x"}
    )
    rows = [
        score(a, "Program: subtract(5, 3)") | {"latency_s": 1.0, "output_tokens": 10},
        score(b, "Program: add(2, 1), divide(#0, 3)") | {"latency_s": 3.0, "output_tokens": 20},
        score(b, "Program: add(1, 3)") | {"latency_s": 2.0, "output_tokens": 30},
        score(a, "garbage") | {"latency_s": 4.0, "output_tokens": 40},
    ]
    assert [r["correct"] for r in rows] == [True, True, False, False]
    assert [r["program_match"] for r in rows] == [True, True, False, False]
    assert [r["valid"] for r in rows] == [True, True, True, False]
    assert [r["official_exec"] for r in rows] == [True, True, False, False]
    assert [r["official_prog"] for r in rows] == [True, True, False, False]
    s = summarize(rows, unseen_reports={"B/1"})
    assert s["exec_acc"] == 0.5 and s["valid_rate"] == 0.75 and s["program_acc"] == 0.5
    assert s["by_steps"]["1"] == 0.5 and s["by_steps"]["2"] == 0.5
    assert math.isnan(s["by_steps"]["3+"])
    assert s["by_evidence"]["text"] == 0.5
    assert s["unseen_report"] == {"n": 2, "exec_acc": 0.5}
    assert s["latency_p50_s"] == 2.5 and s["tokens_per_answer"] == 25
