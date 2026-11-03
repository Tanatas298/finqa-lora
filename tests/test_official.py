from finqa_ft import official

TABLE = [["", "2010", "2009"], ["revenue", "$ 5", "$ 3"], ["costs", "( 2 )", "1"]]


def test_tokens_match_official_format():
    assert official.tokens("subtract(5829, 5735), divide(#0, 5735)") == [
        "subtract(",
        "5829",
        "5735",
        ")",
        "divide(",
        "#0",
        "5735",
        ")",
        "EOF",
    ]
    assert official.tokens(None) == ["EOF"]


def test_execute_rounds_to_five_decimals():
    assert official.execute("divide(1, 3)", TABLE) == 0.33333
    assert official.execute("table_sum(revenue, none)", TABLE) == 8.0
    assert official.execute("greater(5, 3)", TABLE) == "yes"


def test_execute_is_stricter_than_our_interpreter():
    assert official.execute("table_sum(Revenue, none)", TABLE) is None  # case-sensitive rows
    assert official.execute("table_sum(costs, none)", TABLE) is None  # '( 2 )' cell is invalid
    assert official.execute("add($ 5, 3)", TABLE) is None  # '$' in arguments
    assert official.execute("multiply(divide(1, 2), 100)", TABLE) is None  # nested call
    assert official.execute("", TABLE) is None


def test_score_exact_match_and_symbolic_program():
    gold = "subtract(5, 3), divide(#0, 3)"
    s = official.score("subtract(5, 3), divide(#0, 3)", gold, 0.66667, TABLE)
    assert s == {"official_pred": 0.66667, "official_exec": True, "official_prog": True}
    assert official.score("divide(2, 3)", gold, 0.66667, TABLE)["official_exec"]
    assert not official.score("divide(2, 3)", gold, 0.66667, TABLE)["official_prog"]
    assert official.score("add(3, 5)", "add(5, 3)", 8.0, TABLE)["official_prog"]
    # off by x100 is wrong, and 0.1% relative error is wrong
    assert not official.score("divide(200, 3)", gold, 0.66667, TABLE)["official_exec"]
    assert not official.score("divide(2.001, 3)", gold, 0.66667, TABLE)["official_exec"]
    assert not official.score(None, gold, 0.66667, TABLE)["official_prog"]
