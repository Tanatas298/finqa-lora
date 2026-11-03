import pytest

from finqa_ft.program import (
    ProgramError,
    execute,
    execute_steps,
    n_steps,
    normalize_program,
    parse,
    parse_number,
)

TABLE = [
    ["", "2016", "2015", "2014"],
    ["settlements", "-13 ( 13 )", "-19 ( 19 )", "-2 ( 2 )"],
    ["margin", "21.0% ( 21.0 % )", "19.8% ( 19.8 % )", "19.6% ( 19.6 % )"],
    ["net income ( loss )", "$ 1,200", "( 300 )", "n/a"],
    ["q1", "1", "2", "3"],
    ["q1", "10", "20", "30"],
]


@pytest.mark.parametrize(
    "program,expected",
    [
        ("subtract(5829, 5735), divide(#0, 5735)", 94 / 5735),
        ("add(1, 2), multiply(#0, const_100)", 300.0),
        ("multiply(83.7, 7.81%), divide(#0, const_2)", 83.7 * 0.0781 / 2),
        ("exp(1.05, const_3)", 1.05**3),
        ("add(const_m1, const_1000000)", 999999.0),
        ("greater(5, 3)", "yes"),
        ("greater(3, 5)", "no"),
        ("table_average(settlements, none)", -34 / 3),
        ("table_max(margin, none)", 0.21),
        ("table_sum(net income ( loss ), none)", 900.0),
        ("table_min(q1, none)", 10.0),
        ("table_max(MARGIN, none), subtract(#0, 19.6%)", 0.014),
    ],
)
def test_execute(program, expected):
    got = execute(program, TABLE)
    assert got == expected if isinstance(expected, str) else got == pytest.approx(expected)


@pytest.mark.parametrize(
    "program",
    [
        "",
        "subtract(5)",
        "foo(1, 2)",
        "divide(1, 0)",
        "add(#1, 2)",
        "add(x, 2)",
        "table_max(missing row, none)",
        "add(1, 2",
        "greater(1, 2), add(#0, 1)",
    ],
)
def test_invalid(program):
    with pytest.raises(ProgramError):
        execute(program, TABLE)


@pytest.mark.parametrize(
    "text,value",
    [
        ("$ 1,234", 1234.0),
        ("( 12.5 )", -12.5),
        ("7.81%", 0.0781),
        ("const_m1", -1.0),
        ("const_1000", 1000.0),
        ("$ -6129 ( 6129 )", -6129.0),
        ("n/a", None),
        ("const_x", None),
    ],
)
def test_parse_number(text, value):
    assert parse_number(text) == (pytest.approx(value) if value is not None else None)


def test_parse_row_name_with_parens_and_comma():
    assert parse("table_sum(net income ( loss ), none), divide(#0, 2)") == [
        ("table_sum", "net income ( loss )", "none"),
        ("divide", "#0", "2"),
    ]
    assert parse("table_max(sales, net, none)")[0][1] == "sales, net"


def test_execute_steps_trace():
    assert execute_steps("add(1, 2), multiply(#0, 2), greater(#1, 5)") == [3.0, 6.0, "yes"]


def test_normalize_program():
    assert normalize_program("add(3.0, 2)") == normalize_program("add(2, 3)")
    assert normalize_program("subtract(3, 2)") != normalize_program("subtract(2, 3)")
    assert normalize_program("divide(5%,const_100)") == "divide(0.05, const_100)"
    assert n_steps("add(1, 2), add(#0, 3)") == 2
