"""Re-score raw model outputs and write public prediction files (no dataset text).

Reads reports/predictions/<run>.jsonl (raw generations, not committed) and writes
reports/scored/<run>.jsonl with the extracted program, both executors' values and all scores.
"""

import argparse
import json
from pathlib import Path

from finqa_ft.data import load_split
from finqa_ft.metrics import score

KEEP = ("output_tokens", "prompt_tokens", "hit_max_tokens", "latency_s")
DROP = ("gold", "evidence", "report", "steps")


def rescore(path: Path, exs: dict) -> tuple[list[dict], int]:
    rows, changed = [], 0
    for old in map(json.loads, path.read_text().splitlines()):
        new = score(exs[old["id"]], old["output"])
        changed += new["program"] != old["program"]
        new = {k: v for k, v in new.items() if k not in DROP}
        rows.append(new | {k: old[k] for k in KEEP if k in old})
    return rows, changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("reports/predictions"))
    ap.add_argument("--out", type=Path, default=Path("reports/scored"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {s: {e.id: e for e in load_split(s)} for s in ("validation", "test")}
    for path in sorted(args.src.glob("*.jsonl")):
        split = "test" if "_test" in path.stem else "validation"
        rows, changed = rescore(path, splits[split])
        (args.out / path.name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        n = len(rows)
        print(
            f"{path.stem:45s} n={n:5d} official={sum(r['official_exec'] for r in rows) / n:.4f} "
            f"lenient={sum(r['correct'] for r in rows) / n:.4f} parser changes={changed}"
        )


if __name__ == "__main__":
    main()
