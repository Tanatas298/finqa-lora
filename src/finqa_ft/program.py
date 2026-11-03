import math
import re

MATH_OPS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "exp": lambda a, b: a**b,
    "greater": lambda a, b: "yes" if a > b else "no",
}
TABLE_OPS = {
    "table_max": max,
    "table_min": min,
    "table_sum": sum,
    "table_average": lambda xs: sum(xs) / len(xs),
}
OPS = set(MATH_OPS) | set(TABLE_OPS)
_STEP_SPLIT = re.compile(r"\)\s*,\s*(?=(?:" + "|".join(sorted(OPS)) + r")\()")
_STEP = re.compile(r"^\s*([a-z_]+)\((.*)\)\s*$", re.S)

Value = float | str


class ProgramError(ValueError):
    pass


def parse(program: str) -> list[tuple[str, str, str]]:
    """Split a FinQA program into (op, arg1, arg2) steps."""
    parts = _STEP_SPLIT.split(program.strip())
    steps = []
    for i, part in enumerate(parts):
        m = _STEP.match(part if i == len(parts) - 1 else part + ")")
        if not m or m.group(1) not in OPS:
            raise ProgramError(f"bad step: {part!r}")
        args = m.group(2).rsplit(",", 1)
        if len(args) != 2:
            raise ProgramError(f"need two arguments: {part!r}")
        steps.append((m.group(1), args[0].strip(), args[1].strip()))
    return steps


def parse_number(text: str) -> float | None:
    """Parse a FinQA number or table cell: '$ 1,234', '( 12.5 )', '7.81%', 'const_m1'."""
    t = text.strip().lower()
    if t.startswith("const_"):
        rest = t[6:]
        return -1.0 if rest == "m1" else float(rest) if rest.isdigit() else None
    t = re.sub(r"^(.*?\d.*?)\s*\(.*\)$", r"\1", t)  # FinQA cells repeat values: '-13 ( 13 )'
    neg = t.startswith("(") and t.endswith(")")
    t = re.sub(r"[\s$,()]", "", t)
    pct = t.endswith("%")
    t = t.rstrip("%")
    try:
        v = float(t)
    except ValueError:
        return None
    if pct:
        v /= 100
    return -v if neg else v


def _row_values(table: list[list[str]], name: str) -> list[float]:
    key = name.strip().lower()
    for row in reversed(table):  # duplicate row names: gold programs use the last one
        if row and row[0].strip().lower() == key:
            vals = [v for v in (parse_number(c) for c in row[1:]) if v is not None]
            if vals:
                return vals
    raise ProgramError(f"no numeric table row {name!r}")


def execute(program: str, table: list[list[str]] | None = None) -> Value:
    return execute_steps(program, table)[-1]


def execute_steps(program: str, table: list[list[str]] | None = None) -> list[Value]:
    results: list[Value] = []

    def arg(a: str) -> float:
        if a.startswith("#"):
            idx = int(a[1:]) if a[1:].isdigit() else -1
            if not 0 <= idx < len(results):
                raise ProgramError(f"bad reference {a}")
            v = results[idx]
        else:
            v = parse_number(a)
        if not isinstance(v, float):
            raise ProgramError(f"not a number: {a!r}")
        return v

    for op, a1, a2 in parse(program):
        if op in TABLE_OPS:
            results.append(float(TABLE_OPS[op](_row_values(table or [], a1))))
            continue
        try:
            r = MATH_OPS[op](arg(a1), arg(a2))
        except (ZeroDivisionError, OverflowError) as e:
            raise ProgramError(str(e)) from e
        if isinstance(r, complex) or (isinstance(r, float) and not math.isfinite(r)):
            raise ProgramError("non-finite result")
        results.append(r if isinstance(r, str) else float(r))
    return results


def normalize_program(program: str) -> str:
    """Canonical text form; add/multiply operands are order-insensitive."""
    out = []
    for op, a1, a2 in parse(program):
        a1, a2 = _canon_arg(a1), _canon_arg(a2)
        if op in ("add", "multiply"):
            a1, a2 = sorted((a1, a2))
        out.append(f"{op}({a1}, {a2})")
    return ", ".join(out)


def _canon_arg(a: str) -> str:
    if a.startswith("#") or a.startswith("const_") or a == "none":
        return a
    v = parse_number(a)
    return a.lower() if v is None else f"{v:.10g}"


def n_steps(program: str) -> int:
    return len(parse(program))
