import math
import re
from collections import Counter

from finqa_ft.data import Example


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class Retriever:
    """TF-IDF cosine similarity over questions of a pool of (training) examples."""

    def __init__(self, pool: list[Example]):
        self.pool = pool
        docs = [Counter(_tokens(e.question)) for e in pool]
        df = Counter(t for d in docs for t in d)
        self.idf = {t: math.log((1 + len(docs)) / (1 + n)) + 1 for t, n in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts: Counter) -> dict[str, float]:
        v = {t: c * self.idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def top_k(self, question: str, k: int) -> list[Example]:
        q = self._vec(Counter(_tokens(question)))
        scores = [sum(w * v.get(t, 0.0) for t, w in q.items()) for v in self.vecs]
        best = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [self.pool[i] for i in best]
