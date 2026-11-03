"""Rebuild every results table and chart from the scored predictions (no GPU, no model).

Inputs: reports/scored/*.jsonl, reports/metrics/*.json, reports/error_labels.csv, data/raw.
Outputs: docs/results.md, the README tables, reports/*.png and reports/analysis.json.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import charts
import numpy as np

from finqa_ft.analysis import decompose, unseen_mask, wilson, x100_flips
from finqa_ft.data import load_split
from finqa_ft.program import execute_steps, parse
from finqa_ft.retrieval import Retriever
from finqa_ft.stats import ci, mcnemar_exact, paired_diff, resample_idx
from finqa_ft.tables import to_markdown

REPORTS = Path("reports")
SCORED = REPORTS / "scored"
LORA = "qwen3-4b-lora-full-ep2"
SEED1 = "qwen3-4b-lora-full-ep2-seed1"
NAMES = {  # tag: display name
    "qwen3-4b-zeroshot": "Qwen3-4B, zero-shot",
    "qwen3-4b-fewshot": "Qwen3-4B, prompted",
    "qwen3-4b-fewshot-guided": "Qwen3-4B, prompted + guided",
    "qwen3-8b-fewshot": "Qwen3-8B, prompted",
    "qwen3-14b-awq-fewshot": "Qwen3-14B (4-bit), prompted",
    "qwen3-14b-awq-fewshot-guided": "Qwen3-14B (4-bit), prompted + guided",
    "qwen3-4b-lora-n500-ep2": "Qwen3-4B + LoRA, 500 examples",
    "qwen3-4b-lora-n1500-ep2": "Qwen3-4B + LoRA, 1,500 examples",
    LORA: "Qwen3-4B + LoRA, run 1",
    SEED1: "Qwen3-4B + LoRA, run 2",
}
MAIN = [  # rows of the main tables
    "qwen3-4b-zeroshot",
    "qwen3-4b-fewshot",
    "qwen3-4b-fewshot-guided",
    "qwen3-8b-fewshot",
    "qwen3-14b-awq-fewshot",
    "qwen3-14b-awq-fewshot-guided",
    LORA,
    SEED1,
]
README_ROWS = MAIN[1:]
PAIRS = [  # (a, b): significance of a minus b
    (LORA, "qwen3-4b-fewshot"),
    (LORA, "qwen3-4b-fewshot-guided"),
    (LORA, "qwen3-14b-awq-fewshot"),
    (LORA, "qwen3-14b-awq-fewshot-guided"),
    (SEED1, "qwen3-14b-awq-fewshot"),
    (SEED1, "qwen3-14b-awq-fewshot-guided"),
    ("qwen3-4b-fewshot-guided", "qwen3-4b-fewshot"),
    ("qwen3-14b-awq-fewshot-guided", "qwen3-14b-awq-fewshot"),
]
GUIDED = ["qwen3-4b-fewshot-guided", "qwen3-14b-awq-fewshot-guided"]
CURVE = [(500, "qwen3-4b-lora-n500-ep2"), (1500, "qwen3-4b-lora-n1500-ep2"), (6251, LORA)]
VAL_RUNS = {  # validation runs behind each training choice
    "qwen3-4b-lora-n1500-lr1e-4": ("Learning rate", "1e-4 (1,500 examples, 1 epoch)"),
    "qwen3-4b-lora-n1500-lr2e-4": ("Learning rate", "2e-4 (1,500 examples, 1 epoch)"),
    "qwen3-4b-lora-full-ep1": ("Epochs", "1 (all data)"),
    LORA: ("Epochs / run", "2 (all data), run 1"),
    SEED1: ("Run", "2 (all data), run 2"),
}
EXAMPLE = "LMT/2015/page_89.pdf-2"
CATEGORIES = [
    "wrong number extracted",
    "wrong operation",
    "wrong operand order",
    "missing step",
    "scale or percent error",
    "table lookup error",
    "wrong or ambiguous gold label",
    "other",
]


# formatting


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def with_ci(v: tuple[float, float, float]) -> str:
    return f"{100 * v[0]:.1f}% ({100 * v[1]:.1f}–{100 * v[2]:.1f})"


def diff_ci(v: tuple[float, float, float]) -> str:
    return f"{100 * v[0]:+.1f} ({100 * v[1]:+.1f} to {100 * v[2]:+.1f})"


def p_value(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def table(head: list[str], rows: list[list], bold: set[int] = frozenset()) -> str:
    def cell(i: int, v) -> str:
        return f"**{v}**" if i in bold and v != "" else str(v)

    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in enumerate(rows):
        lines.append("| " + " | ".join(cell(i, v) for v in r) + " |")
    return "\n".join(lines)


def bold_rows(tags: list[str]) -> set[int]:
    return {i for i, t in enumerate(tags) if t in (LORA, SEED1)}


# data


def load_scored(split: str = "test") -> dict[str, list[dict]]:
    out = {}
    for tag in NAMES:
        path = SCORED / f"{tag}_{split}.jsonl"
        if path.exists():
            out[tag] = [json.loads(x) for x in path.read_text().splitlines()]
    return out


def col(rows: list[dict], key: str) -> np.ndarray:
    if key == "official_valid":
        return np.array([r["official_pred"] is not None for r in rows])
    return np.array([bool(r[key]) for r in rows])


def summarize(rows: list[dict], idx, unseen, m: dict) -> dict:
    ex, valid = col(rows, "official_exec"), col(rows, "official_valid")
    return {
        "official_exec": ci(ex, idx),
        "official_prog": float(col(rows, "official_prog").mean()),
        "lenient_exec": float(col(rows, "correct").mean()),
        "strict_exec": float(col(rows, "correct_strict").mean()),
        "x100_flips": x100_flips(rows),
        "valid": float(valid.mean()),
        "acc_valid": float(ex[valid].mean()),
        "unseen": ci(ex[unseen], resample_idx(int(unseen.sum()))),
        "seen": ci(ex[~unseen], resample_idx(int((~unseen).sum()))),
        "latency_p50_s": m.get("latency_p50_s"),
        "tokens_per_answer": float(np.mean([r["output_tokens"] for r in rows])),
        "weights_gb": m.get("weights_gb"),
        "power_w": m.get("mean_power_w"),
        "throughput": m.get("throughput_q_per_s"),
        "cost_eur_per_1000q": m.get("cost_eur_per_1000q"),
    }


def significance(scored, idx) -> list[dict]:
    out = []
    for a, b in PAIRS:
        xa, xb = col(scored[a], "official_exec"), col(scored[b], "official_exec")
        out.append({"a": a, "b": b, "diff": paired_diff(xa, xb, idx), **mcnemar_exact(xa, xb)})
    return out


def leakage(train: list, test: list) -> dict:
    retriever = Retriever(train)
    reports = {e.report for e in train}
    tickers = {e.report.split("/")[0] for e in train}
    return {
        "test_company_years_in_train": len({e.report for e in test} & reports),
        "unseen_company_year_questions": sum(e.report not in reports for e in test),
        "unseen_company_questions": sum(e.report.split("/")[0] not in tickers for e in test),
        "shot_from_same_company_year": sum(
            any(s.report == e.report for s in retriever.top_k(e.question, 4)) for e in test
        ),
    }


def errors(scored) -> dict:
    labels = list(csv.DictReader((REPORTS / "error_labels.csv").open(encoding="utf-8")))
    bad = [r["id"] for r in labels if r["category"] == "wrong or ambiguous gold label"]
    by_id = {t: {r["id"]: r for r in rows} for t, rows in scored.items()}
    return {
        "n": len(labels),
        "counts": Counter(r["category"] for r in labels),
        "wrong_gold": sum(r["note"].startswith("wrong gold") for r in labels),
        "bad_gold_ids": bad,
        "noise": {t: sum(by_id[t][i]["official_exec"] for i in bad) for t in scored},
        "n_fail": sum(not r["official_exec"] for r in scored[LORA]),
        "wilson": wilson(len(bad), len(labels)),
    }


def residual_cause(r: dict) -> str:
    if r["hit_max_tokens"]:
        return "cut off at the token limit"
    if r["valid"]:
        return "table cell the official script cannot parse"
    return "table row without numbers"


def validation_runs() -> dict:
    out = {}
    for tag in VAL_RUNS:
        path = SCORED / f"{tag}_validation.jsonl"
        if path.exists():
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            out[tag] = float(col(rows, "official_exec").mean())
    return out


# README blocks


def readme_results(s: dict) -> str:
    rows = [
        [
            NAMES[t],
            pct(s[t]["official_exec"][0]),
            pct(s[t]["official_prog"]),
            pct(s[t]["valid"]),
            f"{s[t]['latency_p50_s']:.2f} s",
            f"{s[t]['weights_gb']:.0f} GB",
        ]
        for t in README_ROWS
    ]
    head = ["Model", "Accuracy", "Program match", "Valid output", "Latency", "Size"]
    return table(head, rows, bold_rows(README_ROWS))


def readme_runs(s: dict) -> str:
    keys = [
        ("Accuracy", "official_exec"),
        ("Program match", "official_prog"),
        ("Accuracy on reports unseen in training", "unseen"),
    ]
    rows = []
    for label, k in keys:
        a, b = s[LORA][k], s[SEED1][k]
        a, b = (a[0], b[0]) if isinstance(a, tuple) else (a, b)
        rows.append([label, pct(a), pct(b), pct((a + b) / 2)])
    return table(["", "Run 1", "Run 2", "Mean"], rows)


def readme_curve(s: dict) -> str:
    return table(
        ["Training examples"] + [f"{n:,}" for n, _ in CURVE],
        [["Accuracy"] + [pct(s[t]["official_exec"][0]) for _, t in CURVE]],
    )


def worked_example(ex, scored) -> str:
    lora = {r["id"]: r for r in scored[LORA]}[ex.id]
    steps = parse(lora["program"])
    values = execute_steps(lora["program"], ex.table)
    used = {a for _, a, b in steps for a in (a, b)}
    rows = [ex.table[0]] + [r for r in ex.table[1:] if used & {c.strip() for c in r[1:]}]
    trace = "\n".join(
        f"#{i} = {op}({a}, {b}) = {v:.4g}"
        for i, ((op, a, b), v) in enumerate(zip(steps, values, strict=True))
    )
    return (
        f"> {ex.question}\n\n{to_markdown(rows)}\n\n"
        f"```text\nProgram: {lora['program']}\n{trace}\n```"
    )


# docs/results.md


def results_md(
    s: dict, sig: list, lk: dict, err: dict, val: dict, n: int, exs, scored, run: dict
) -> str:
    n_un = lk["unseen_company_year_questions"]
    parts = [
        "# Results",
        "All numbers on the FinQA test set (1,147 questions), scored with FinQA's official "
        "evaluation script, unless marked otherwise. Ranges in parentheses are 95% confidence "
        "intervals from a paired bootstrap (10,000 resamples). How each number is measured: "
        "[evaluation.md](evaluation.md). Generated by `scripts/make_tables.py`; do not edit by "
        "hand.",
        "## Contents",
        "1. [Main results](#main-results)\n2. [Significance](#significance)\n"
        "3. [Training runs and choices](#training-runs-and-choices)\n"
        "4. [Training data size](#training-data-size)\n"
        "5. [Unseen reports](#unseen-reports)\n6. [Where the gain comes from]"
        "(#where-the-gain-comes-from)\n7. [Error analysis](#error-analysis)\n"
        "8. [Official vs lenient scoring](#official-vs-lenient-scoring)\n"
        "9. [Cost](#cost)",
    ]

    rows = [
        [
            NAMES[t],
            with_ci(s[t]["official_exec"]),
            pct(s[t]["official_prog"]),
            pct(s[t]["valid"]),
            f"{s[t]['latency_p50_s']:.2f} s",
            f"{s[t]['tokens_per_answer']:.0f}",
        ]
        for t in MAIN
    ]
    head = ["Model", "Accuracy", "Program match", "Valid output", "Latency", "Tokens"]
    parts += [
        "## Main results",
        table(head, rows, bold_rows(MAIN)),
        "*Prompted*: 4 similar solved training examples in the prompt. *Guided*: decoding "
        "restricted to valid programs. *Latency*: median seconds per question, batch size 1. "
        "*Tokens*: mean output length.",
    ]

    rows = [
        [f"{NAMES[r['a']]} vs {NAMES[r['b']]}", diff_ci(r["diff"]), p_value(r["p"])] for r in sig
    ]
    parts += [
        "## Significance",
        "Accuracy difference in percentage points, paired over the same questions; p-value "
        "from an exact McNemar test.",
        table(["Comparison", "Difference", "p-value"], rows),
    ]

    rows = [[what, value, pct(val[t])] for t, (what, value) in VAL_RUNS.items() if t in val]
    keys = [
        ("Accuracy", "official_exec"),
        ("Program match", "official_prog"),
        ("Valid output", "valid"),
        ("Accuracy, unseen reports", "unseen"),
    ]
    seed_rows = []
    for label, k in keys:
        a, b = s[LORA][k], s[SEED1][k]
        a, b = (a[0], b[0]) if isinstance(a, tuple) else (a, b)
        seed_rows.append([label, pct(a), pct(b), pct((a + b) / 2)])
    parts += [
        "## Training runs and choices",
        "The final configuration (all 6,251 examples, 2 epochs, learning rate 2e-4) was trained "
        "twice with different seeds. With two runs both values are shown; no standard deviation "
        "is claimed.",
        table(["Test set", "Run 1", "Run 2", "Mean"], seed_rows),
        "Every training choice was made on the validation set (883 questions):",
        table(["Choice", "Setting", "Validation accuracy"], rows),
        "Run 1 is the model on the Hugging Face Hub because it is better on validation; run 2 is "
        "better on test, which played no part in the choice. Run 1 was resumed from step 500 "
        "after an interruption, so its logged training loss covers only the last steps and is "
        "not comparable with run 2's.",
    ]

    rows = [[f"{k:,}", with_ci(s[t]["official_exec"]), pct(s[t]["valid"])] for k, t in CURVE]
    parts += [
        "## Training data size",
        "Subsets are prefixes of one shuffled training set (the 500 examples are inside the "
        "1,500), trained with the same settings as the final model.",
        table(["Training examples", "Accuracy", "Valid output"], rows),
        f"For reference: Qwen3-14B prompted {pct(s['qwen3-14b-awq-fewshot']['official_exec'][0])},"
        f" prompted + guided {pct(s['qwen3-14b-awq-fewshot-guided']['official_exec'][0])}.",
    ]

    rows = [[NAMES[t], with_ci(s[t]["unseen"]), with_ci(s[t]["seen"])] for t in MAIN]
    parts += [
        "## Unseen reports",
        f"FinQA's splits share no report pages, but {lk['test_company_years_in_train']} annual "
        f"reports (company and year) appear in both train and test. {n_un} test questions come "
        "from reports not in the training set.",
        table(
            ["Model", f"Unseen reports ({n_un})", f"Seen reports ({n - n_un:,})"],
            rows,
            bold_rows(MAIN),
        ),
        f"Only {lk['unseen_company_questions']} test questions come from a company that is not in "
        "the training set at all, too few to score. For "
        f"{lk['shot_from_same_company_year']} test questions "
        f"({lk['shot_from_same_company_year'] / n:.1%}) at least one prompt example comes from "
        "the same report.",
    ]

    b, f = s["qwen3-4b-fewshot"], s[LORA]
    d = decompose(b, f)
    evid = Counter(exs[i].evidence for i in exs)
    ev_rows = []
    for t in MAIN:
        vals = []
        for k in ("table", "text", "both"):
            sel = [r["official_exec"] for r in scored[t] if exs[r["id"]].evidence == k]
            vals.append(pct(sum(sel) / len(sel)))
        ev_rows.append([NAMES[t], *vals])
    parts += [
        "## Where the gain comes from",
        "Accuracy = share of valid programs × accuracy among valid programs.",
        table(
            ["Model", "Valid output", "Accuracy among valid", "Accuracy"],
            [
                [
                    NAMES[t],
                    pct(s[t]["valid"]),
                    pct(s[t]["acc_valid"]),
                    pct(s[t]["official_exec"][0]),
                ]
                for t in ("qwen3-4b-fewshot", "qwen3-4b-fewshot-guided", LORA)
            ],
            {2},
        ),
        f"Fine-tuning adds {100 * d['total']:.1f} points over the prompted 4B model: "
        f"{100 * d['format']:.1f} from producing valid programs and {100 * d['correctness']:.1f} "
        "from getting more of them right (each factor's change weighted by the average of the "
        "other factor). Guided decoding fixes the format of the prompted model but adds only "
        f"{100 * (s['qwen3-4b-fewshot-guided']['official_exec'][0] - b['official_exec'][0]):.1f}"
        " points: most of its invalid programs were on questions it would have got wrong anyway.",
        "Accuracy by where the needed numbers are:",
        table(
            [
                "Model",
                f"Table only ({evid['table']})",
                f"Text only ({evid['text']})",
                f"Both ({evid['both']})",
            ],
            ev_rows,
            bold_rows(MAIN),
        ),
    ]

    lo, hi = err["wilson"]
    k = len(err["bad_gold_ids"])
    noise_rows = [[NAMES[t], f"{err['noise'][t]} / {k}"] for t in MAIN]
    parts += [
        "## Error analysis",
        f"{err['n']} failures of run 1, sampled with a fixed seed and labelled by hand "
        "([error_labels.csv](../reports/error_labels.csv)).",
        table(["Category", "Count"], [[c, err["counts"].get(c, 0)] for c in CATEGORIES]),
        f"A gold label counts as wrong or ambiguous only when the model's own answer is "
        f"defensible ({err['wrong_gold']} wrong, {k - err['wrong_gold']} ambiguous). Extrapolated "
        f"to all {err['n_fail']} failures, about {lo * err['n_fail'] / n:.0%}–"
        f"{hi * err['n_fail'] / n:.0%} of the test set has such a label. Other models rarely "
        "answer these questions correctly either:",
        table(["Model", f"Correct on the {k} questions"], noise_rows),
    ]

    rows = [
        [
            NAMES[t],
            pct(s[t]["official_exec"][0]),
            pct(s[t]["lenient_exec"]),
            str(s[t]["x100_flips"]),
        ]
        for t in MAIN
    ]
    res_rows = []
    for t in GUIDED:
        bad = [r for r in scored[t] if r["official_pred"] is None]
        causes = Counter(residual_cause(r) for r in bad)
        res_rows.append(
            [NAMES[t], str(len(bad)), "; ".join(f"{c} ({v})" for c, v in causes.items())]
        )
    parts += [
        "## Official vs lenient scoring",
        "The lenient metric (0.1% tolerance, answers off by exactly ×100 count as correct) was "
        "the project's original score; the official script requires an exact match after "
        "rounding to 5 decimals.",
        table(["Model", "Official", "Lenient", "Saved by the ×100 rule"], rows, bold_rows(MAIN)),
        "Invalid programs left under guided decoding:",
        table(["Model", "Invalid", "Cause"], res_rows),
    ]

    rows = [
        [
            NAMES[t],
            f"{s[t]['weights_gb']:.1f} GB",
            f"{s[t]['throughput']:.1f}",
            f"€{s[t]['cost_eur_per_1000q']:.3f}",
        ]
        for t in MAIN
        if s[t]["throughput"]
    ]
    parts += [
        "## Cost",
        "Batched inference on one 24 GB GPU. Energy cost = measured board power × time × "
        "€0.30/kWh, GPU only.",
        table(["Model", "Weights", "Questions per second", "Energy per 1,000 questions"], rows),
        f"Training the final model: {run['steps']} steps, "
        f"{run['wall_seconds'] / 3600:.1f} hours, peak {run['peak_vram_gb']:.1f} GB of GPU "
        "memory (run 2, which ran without interruption).",
    ]
    return "\n\n".join(parts) + "\n"


def final_run() -> dict:
    """Training log of the uninterrupted final run (all data, 2 epochs, seed 1)."""
    runs = [json.loads(x) for x in (REPORTS / "train_runs.jsonl").read_text().splitlines()]
    return next(r for r in runs if r["seed"] == 1 and r["train_limit"] is None and r["epochs"] == 2)


def fill(path: Path, blocks: dict[str, str]) -> None:
    """Replace the text between <!-- BEGIN name --> and <!-- END name --> markers."""
    text = path.read_text(encoding="utf-8")
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)", re.S)
        text, n = pat.subn(lambda m, b=body: m.group(1) + b.strip("\n") + "\n" + m.group(2), text)
        assert n == 1, f"marker {name} missing in {path}"
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    train, test = load_split("train"), load_split("test")
    exs = {e.id: e for e in test}
    scored = load_scored()
    ids = [r["id"] for r in scored[LORA]]
    assert all([r["id"] for r in rows] == ids for rows in scored.values())
    idx = resample_idx(len(ids))
    unseen = unseen_mask(ids, exs, {e.report for e in train})
    s = {}
    for tag, rows in scored.items():
        p = REPORTS / "metrics" / f"{tag}_test.json"
        s[tag] = summarize(rows, idx, unseen, json.loads(p.read_text()) if p.exists() else {})

    sig = significance(scored, idx)
    lk = leakage(train, test)
    err = errors(scored)
    val = validation_runs()

    by = {t: {"tag": t, **v} for t, v in s.items()}
    charts.size_chart(by, REPORTS / "acc_vs_size.png")
    charts.curve_chart(by, CURVE, REPORTS / "data_efficiency.png")
    charts.latency_chart(by, REPORTS / "acc_vs_latency.png")
    fill(
        Path("README.md"),
        {
            "results": readme_results(s),
            "runs": readme_runs(s),
            "curve": readme_curve(s),
            "example": worked_example(exs[EXAMPLE], scored),
        },
    )
    md = results_md(s, sig, lk, err, val, len(ids), exs, scored, final_run())
    Path("docs/results.md").write_text(md, encoding="utf-8", newline="\n")
    analysis = {
        "n_test": len(ids),
        "models": s,
        "significance": sig,
        "leakage": lk,
        "errors": err,
        "validation": val,
    }
    (REPORTS / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", newline="\n")
    print("wrote docs/results.md, README tables, charts, reports/analysis.json")


if __name__ == "__main__":
    main()
