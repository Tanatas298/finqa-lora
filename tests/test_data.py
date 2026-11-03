from conftest import make

from finqa_ft.data import overlap


def test_overlap_detects_shared_pages_and_reports():
    splits = {
        "train": [make("A/2010/page_1.pdf-1"), make("B/2011/page_5.pdf-2")],
        "test": [make("A/2010/page_9.pdf-1"), make("C/2012/page_1.pdf-1")],
        "validation": [make("B/2011/page_5.pdf-3")],
    }
    assert overlap(splits) == {
        "test&train": set(),
        "test&validation": set(),
        "train&validation": {"B/2011/page_5.pdf"},
    }
    assert overlap(splits, "report")["test&train"] == {"A/2010"}


def test_evidence_and_context(ex):
    assert ex.evidence == "table"
    assert make(gold_inds={"text_1": "x", "table_2": "y"}).evidence == "both"
    assert make(gold_inds={"text_0": "x"}).evidence == "text"
    assert "| revenue | $ 5 | $ 3 |" in ex.context()
    assert ex.report == "A/2010"
