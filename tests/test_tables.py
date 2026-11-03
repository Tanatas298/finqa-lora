from finqa_ft.tables import to_markdown


def test_to_markdown():
    md = to_markdown([["", "2010"], ["a|b", "$ 5", "extra"]])
    assert md.splitlines() == [
        "|  | 2010 |  |",
        "|---|---|---|",
        "| a\\|b | $ 5 | extra |",
    ]


def test_empty():
    assert to_markdown([]) == ""
