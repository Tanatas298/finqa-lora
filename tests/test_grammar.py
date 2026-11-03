import re

from finqa_ft.grammar import program_regex, row_names

TABLE = [["", "2010", "2009"], ["revenue", "$ 5", "$ 3"], ["costs ( net )", "2", "1"]]


def ok(text: str) -> bool:
    return re.fullmatch(program_regex(TABLE), text) is not None


def test_row_names_skip_header():
    assert row_names(TABLE) == ["costs ( net )", "revenue"]


def test_accepts_valid_programs():
    assert ok("Program: subtract(5829, 5735), divide(#0, 5735)")
    assert ok("Program: divide(-1.5, .76), multiply(#0, const_100), add(#1, 7.81%)")
    assert ok("Program: table_sum(revenue, none), divide(#0, const_m1)")
    assert ok("Program: table_max(costs ( net ), none)")


def test_rejects_invalid_programs():
    assert not ok("subtract(5, 3)")  # missing prefix
    assert not ok("Program: divide(#0, 2)")  # forward reference
    assert not ok("Program: add(1, 2), add(#1, 3)")  # reference to the current step
    assert not ok("Program: multiply(divide(1, 2), 100)")  # nested call
    assert not ok("Program: table_sum(profit, none)")  # row not in table
    assert not ok("Program: round(1.5, 0)")  # unknown op
    assert not ok("Program: add(1,2)")  # official tokenizer needs ', '
    assert not ok("Program: " + ", ".join(["add(1, 2)"] * 9))  # more than 8 steps
