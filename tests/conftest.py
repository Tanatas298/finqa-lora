import pytest

from finqa_ft.data import Example


def make(
    id: str = "A/2010/page_1.pdf-1",
    question: str = "what is the change in revenue?",
    program: str = "subtract(5, 3)",
    answer: float | str = 2.0,
    table: list[list[str]] | None = None,
    gold_inds: dict[str, str] | None = None,
) -> Example:
    return Example(
        id=id,
        filename=id.rsplit("-", 1)[0],
        question=question,
        pre_text=["revenue grew ."],
        post_text=["end ."],
        table=table or [["", "2010", "2009"], ["revenue", "$ 5", "$ 3"]],
        program=program,
        answer=answer,
        gold_inds=gold_inds or {"table_1": "revenue is 5 and 3"},
    )


@pytest.fixture
def ex() -> Example:
    return make()
