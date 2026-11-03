import statistics

from finqa_ft import official
from finqa_ft.data import Example
from finqa_ft.normalize import answers_match
from finqa_ft.program import ProgramError, execute, n_steps, normalize_program
from finqa_ft.prompts import extract_program


def score(ex: Example, output: str) -> dict:
    program = extract_program(output)
    pred, valid, prog_match = None, False, False
    if program:
        try:
            pred = execute(program, ex.table)
            valid = True
            prog_match = normalize_program(program) == normalize_program(ex.program)
        except (ProgramError, ValueError, TypeError):
            pass
    return {
        "id": ex.id,
        "program": program,
        "pred": pred,
        "gold": ex.answer,
        "valid": valid,
        "correct": answers_match(pred, ex.answer),
        "correct_strict": answers_match(pred, ex.answer, strict=True),
        "program_match": prog_match,
        **official.score(program, ex.program, ex.answer, ex.table),
        "steps": min(n_steps(ex.program), 3),
        "evidence": ex.evidence,
        "report": ex.report,
    }


def _rate(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def _pct(xs: list[float], q: float) -> float:
    if len(xs) < 2:
        return xs[0] if xs else float("nan")
    return statistics.quantiles(xs, n=100, method="inclusive")[round(q) - 1]


def summarize(rows: list[dict], unseen_reports: set[str] | None = None) -> dict:
    out = {
        "n": len(rows),
        "exec_acc": _rate(rows, "correct"),
        "exec_acc_strict": _rate(rows, "correct_strict"),
        "program_acc": _rate(rows, "program_match"),
        "official_exec_acc": _rate(rows, "official_exec"),
        "official_prog_acc": _rate(rows, "official_prog"),
        "valid_rate": _rate(rows, "valid"),
        "by_steps": {
            f"{s}{'+' if s == 3 else ''}": _rate([r for r in rows if r["steps"] == s], "correct")
            for s in (1, 2, 3)
        },
        "by_evidence": {
            e: _rate([r for r in rows if r["evidence"] == e], "correct")
            for e in ("table", "text", "both")
        },
    }
    if unseen_reports is not None:
        unseen = [r for r in rows if r["report"] in unseen_reports]
        out["unseen_report"] = {"n": len(unseen), "exec_acc": _rate(unseen, "correct")}
    lat = [r["latency_s"] for r in rows if "latency_s" in r]
    toks = [r["output_tokens"] for r in rows if "output_tokens" in r]
    if lat:
        out["latency_p50_s"], out["latency_p95_s"] = _pct(lat, 50), _pct(lat, 95)
    if toks:
        out["tokens_per_answer"] = statistics.mean(toks)
    return out
