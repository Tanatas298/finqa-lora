import json
from dataclasses import dataclass
from pathlib import Path

from finqa_ft.tables import to_markdown

RAW_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{}.json"
SPLITS = {"train": "train", "validation": "dev", "test": "test"}
SPLIT_SIZES = {"train": 6251, "validation": 883, "test": 1147}


@dataclass
class Example:
    id: str
    filename: str
    question: str
    pre_text: list[str]
    post_text: list[str]
    table: list[list[str]]
    program: str
    answer: float | str
    gold_inds: dict[str, str]

    @property
    def report(self) -> str:
        """Company/year of the annual report, e.g. 'ETR/2016'."""
        return "/".join(self.filename.split("/")[:2])

    @property
    def evidence(self) -> str:
        kinds = {k.split("_")[0] for k in self.gold_inds}
        return "both" if len(kinds) > 1 else next(iter(kinds), "none")

    def context(self) -> str:
        return (
            f"Text before the table:\n{' '.join(self.pre_text)}\n\n"
            f"Table:\n{to_markdown(self.table)}\n\n"
            f"Text after the table:\n{' '.join(self.post_text)}"
        )

    def gold_evidence(self) -> str:
        return "\n".join(self.gold_inds.values())


def from_raw(d: dict) -> Example:
    qa = d["qa"]
    return Example(
        id=d["id"],
        filename=d["filename"],
        question=qa["question"],
        pre_text=d["pre_text"],
        post_text=d["post_text"],
        table=d["table"],
        program=qa["program"],
        answer=qa["exe_ans"],
        gold_inds=qa.get("gold_inds", {}),
    )


def load_split(split: str, raw_dir: Path = Path("data/raw")) -> list[Example]:
    with open(raw_dir / f"{SPLITS[split]}.json", encoding="utf-8") as f:
        return [from_raw(d) for d in json.load(f)]


def overlap(splits: dict[str, list[Example]], key: str = "filename") -> dict[str, set[str]]:
    """Shared keys ('filename' = report page, 'report' = company/year) for each split pair."""
    keys = {s: {getattr(e, key) for e in exs} for s, exs in splits.items()}
    names = sorted(keys)
    return {f"{a}&{b}": keys[a] & keys[b] for i, a in enumerate(names) for b in names[i + 1 :]}
