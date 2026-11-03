import argparse
import json
import statistics
import urllib.request
from pathlib import Path

from finqa_ft.data import RAW_URL, SPLIT_SIZES, SPLITS, load_split, overlap
from finqa_ft.normalize import answers_match
from finqa_ft.program import ProgramError, execute
from finqa_ft.prompts import messages, render, target


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS.values():
        path = raw_dir / f"{name}.json"
        if not path.exists():
            urllib.request.urlretrieve(RAW_URL.format(name), path)


def gold_check(exs) -> dict:
    mismatches = []
    for e in exs:
        try:
            pred = execute(e.program, e.table)
        except ProgramError as err:
            pred = str(err)
        if not answers_match(pred, e.answer, strict=True):
            mismatches.append({"id": e.id, "program": e.program, "gold": e.answer, "got": pred})
    return {"match_rate": 1 - len(mismatches) / len(exs), "mismatches": mismatches}


def token_stats(splits, tokenizer_name: str) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    out = {}
    for name, exs in splits.items():
        lens = sorted(len(tok(render(tok, messages(e)) + target(e)).input_ids) for e in exs)
        q = statistics.quantiles(lens, n=100)
        out[name] = {
            "mean": statistics.mean(lens),
            "p50": q[49],
            "p95": q[94],
            "p99": q[98],
            "max": lens[-1],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument(
        "--tokenizer", default=None, help="e.g. Qwen/Qwen3-4B; skip token stats if unset"
    )
    ap.add_argument("--out", type=Path, default=Path("reports/data_checks.json"))
    ap.add_argument("--download-only", action="store_true", help="skip the checks")
    args = ap.parse_args()

    download(args.raw_dir)
    if args.download_only:
        return
    splits = {s: load_split(s, args.raw_dir) for s in SPLITS}
    sizes = {s: len(v) for s, v in splits.items()}
    assert sizes == SPLIT_SIZES, sizes
    pages = overlap(splits)
    assert not any(pages.values()), f"report pages shared across splits: {pages}"
    reports = overlap(splits, "report")
    train_reports = {e.report for e in splits["train"]}
    checks = {
        "sizes": sizes,
        "page_overlap": {k: len(v) for k, v in pages.items()},
        "company_year_overlap": {k: len(v) for k, v in reports.items()},
        "test_questions_unseen_company_year": sum(
            e.report not in train_reports for e in splits["test"]
        ),
        "gold_programs": {s: gold_check(v) for s, v in splits.items()},
    }
    if args.tokenizer:
        checks["prompt_tokens"] = token_stats(splits, args.tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(checks, indent=2))
    summary = {k: v for k, v in checks.items() if k != "gold_programs"}
    summary["gold_match"] = {s: v["match_rate"] for s, v in checks["gold_programs"].items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
