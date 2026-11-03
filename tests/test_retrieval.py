from conftest import make

from finqa_ft.retrieval import Retriever


def test_top_k_ranks_similar_questions_first():
    pool = [
        make("A/1/p.pdf-1", "what was the percentage change in net revenue?"),
        make("A/1/p.pdf-2", "what is the total debt maturing in 2012?"),
        make("A/1/p.pdf-3", "what is the average dividend per share?"),
    ]
    r = Retriever(pool)
    assert r.top_k("percentage change in revenue from 2014 to 2015", 2)[0].id == "A/1/p.pdf-1"
    assert r.top_k("debt maturing", 1)[0].id == "A/1/p.pdf-2"
    assert len(r.top_k("unrelated words", 2)) == 2
