"""Adapter around FinQA's official evaluation script (vendored unchanged).

The official script reads {"id", "predicted": tokens} and the raw dataset. Here the generated
program string goes through the script's own `program_tokenization`, `eval_program` and
`equal_program`, so the scoring logic is the official one, applied one question at a time.
"""

import contextlib
import io

from finqa_ft.vendor import finqa_evaluate as official


def tokens(program: str | None) -> list[str]:
    return official.program_tokenization(program or "")


def execute(program: str | None, table: list[list[str]]) -> float | str | None:
    """Official execution result, rounded to 5 decimals as the script does; None if invalid."""
    invalid, res = official.eval_program(tokens(program), table)
    return None if invalid or res == "n/a" else res


def score(program: str | None, gold_program: str, gold_answer: float | str, table) -> dict:
    """Official execution match (exact, after rounding) and program match (symbolic)."""
    res = execute(program, table)
    with contextlib.redirect_stdout(io.StringIO()):  # the script prints on structure errors
        prog = official.equal_program(tokens(gold_program), tokens(program))
    return {
        "official_pred": res,
        "official_exec": res is not None and res == gold_answer,
        "official_prog": bool(prog),
    }
