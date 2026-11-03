import re
from collections.abc import Sequence

from finqa_ft.data import Example
from finqa_ft.program import ProgramError, parse

SYSTEM = """You answer numerical questions about a company's financial report.
Write the calculation as a program in this language:
- Math operations, each with two arguments: add, subtract, multiply, divide, exp (power), \
greater (returns yes/no).
- Table operations over one table row, second argument is always none: table_max, table_min, \
table_sum, table_average. The first argument is the row name exactly as in the first column.
- Arguments are numbers copied from the report, constants (const_100, const_1000, \
const_1000000, const_m1 for -1, const_2, ...), or #n, the result of step n (counting from 0).
- Steps are separated by ", ". Write percentages as written, e.g. 7.81%.
Example: subtract(5829, 5735), divide(#0, 5735)
End your answer with one final line: Program: <program>"""


def user_message(ex: Example, shots: Sequence[Example] = ()) -> str:
    parts = []
    if shots:
        parts.append("Solved examples (supporting facts only):")
        parts += [
            f"Question: {s.question}\nFacts:\n{s.gold_evidence()}\nProgram: {s.program}"
            for s in shots
        ]
        parts.append("Now the real task.")
    parts.append(f"{ex.context()}\n\nQuestion: {ex.question}")
    return "\n\n".join(parts)


def messages(ex: Example, shots: Sequence[Example] = ()) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(ex, shots)},
    ]


def render(tokenizer, msgs: list[dict[str, str]]) -> str:
    """Chat-template prompt text (thinking off), shared by evaluation and training."""
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def target(ex: Example) -> str:
    return f"Program: {ex.program}"


def extract_program(output: str) -> str | None:
    """Take the program from the last 'Program:' line, after removing <think> blocks."""
    text = re.sub(r"<think>.*?(</think>|$)", "", output, flags=re.S)
    lines = [ln.strip().strip("`") for ln in text.strip().splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lower().startswith("program:"):
            rest = lines[i].split(":", 1)[1].strip(" `")
            return rest or (lines[i + 1].strip(" `") if i + 1 < len(lines) else None)
    for ln in reversed(lines):
        try:
            parse(ln)
            return ln
        except ProgramError:
            continue
    return None
