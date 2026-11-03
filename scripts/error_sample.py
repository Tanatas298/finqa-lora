"""Sample model failures for hand labelling into reports/error_labels.csv.

Existing labels for sampled ids are kept, so rerunning never discards hand work.
"""

import argparse
import csv
import json
import random
from pathlib import Path

FIELDS = ["id", "program", "pred", "category", "note"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="qwen3-4b-lora-full-ep2_test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("reports/error_labels.csv"))
    args = ap.parse_args()

    path = Path(f"reports/scored/{args.run}.jsonl")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    fails = sorted((r for r in rows if not r["official_exec"]), key=lambda r: r["id"])
    sample = random.Random(args.seed).sample(fails, args.n)
    old = {}
    if args.out.exists():
        old = {r["id"]: r for r in csv.DictReader(args.out.open(encoding="utf-8"))}
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS, lineterminator="\n")
        w.writeheader()
        for r in sample:
            prev = old.get(r["id"], {})
            w.writerow(
                {
                    "id": r["id"],
                    "program": r["program"],
                    "pred": r["official_pred"] if r["official_pred"] is not None else r["pred"],
                    "category": prev.get("category", ""),
                    "note": prev.get("note", ""),
                }
            )
    print(f"wrote {args.out}: {len(sample)} of {len(fails)} failures")


if __name__ == "__main__":
    main()
