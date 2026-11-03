import argparse
import csv
import json
import random
import urllib.request
from pathlib import Path

from finqa_ft.data import Example, load_split
from finqa_ft.program import ProgramError, execute_steps, parse
from finqa_ft.prompts import extract_program, messages


def ask(url: str, model: str, ex: Example) -> str:
    body = {
        "model": model,
        "messages": messages(ex),
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{url}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the served model a FinQA-style question")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="finqa-4b")
    ap.add_argument("--split", default="test")
    ap.add_argument("--id", help="example id; random example if neither --id nor --question")
    ap.add_argument("--question")
    ap.add_argument("--table", type=Path, help="CSV file, first row is the header")
    ap.add_argument("--text", default="", help="report text around the table")
    args = ap.parse_args()

    if args.question:
        table = list(csv.reader(args.table.open())) if args.table else []
        ex = Example("custom", "custom", args.question, [args.text], [], table, "", "", {})
    else:
        exs = load_split(args.split)
        ex = next(e for e in exs if e.id == args.id) if args.id else random.choice(exs)
        print(f"[{ex.id}]")
    print(f"{ex.context()}\n\nQuestion: {ex.question}\n")

    program = extract_program(ask(args.url, args.model, ex))
    print(f"Program: {program}")
    try:
        for i, ((op, a, b), v) in enumerate(
            zip(parse(program), execute_steps(program, ex.table), strict=True)
        ):
            print(f"  #{i} = {op}({a}, {b}) = {v}")
    except (ProgramError, TypeError) as e:
        print(f"  could not execute: {e}")
        return
    if ex.program:
        print(f"Gold: {ex.program} = {ex.answer}")


if __name__ == "__main__":
    main()
