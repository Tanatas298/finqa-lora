"""Regular grammar for guided decoding: 'Program: ' followed by a syntactically valid program.

Step k may only reference #0..#(k-1). Table operations may only name rows of the question's
table. Numbers are literals (optionally negative, decimal or percent) or FinQA constants.
"""

import re

MAX_STEPS = 8  # longest gold program in train has 6 steps
NUM = r"-?([0-9]+(\.[0-9]+)?|\.[0-9]+)%?"
CONST = r"const_(m1|[0-9]+)"
MATH = "(add|subtract|multiply|divide|exp|greater)"
TABLE = "(table_max|table_min|table_sum|table_average)"


def row_names(table: list[list[str]]) -> list[str]:
    names = {r[0].strip() for r in table[1:] if r and r[0].strip()}
    return sorted(n for n in names if "\n" not in n)


def _escape(name: str) -> str:
    return re.escape(name).replace("\\ ", " ")


def _step(k: int, rows: list[str]) -> str:
    refs = [f"#{i}" for i in range(k)]
    arg = "(" + "|".join([NUM, CONST] + refs) + ")"
    alts = [rf"{MATH}\({arg}, {arg}\)"]
    if rows:
        alts.append(rf"{TABLE}\(({'|'.join(map(_escape, rows))}), none\)")
    return "(" + "|".join(alts) + ")"


def program_regex(table: list[list[str]], max_steps: int = MAX_STEPS) -> str:
    rows = row_names(table)
    tail = ""
    for k in range(max_steps - 1, 0, -1):
        tail = f"(, {_step(k, rows)}{tail})?"
    return f"Program: {_step(0, rows)}{tail}"
