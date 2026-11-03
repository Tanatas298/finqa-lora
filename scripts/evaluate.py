import argparse
import json
import random
import subprocess
import threading
import time
from pathlib import Path

from finqa_ft.data import load_split
from finqa_ft.grammar import program_regex
from finqa_ft.metrics import score, summarize
from finqa_ft.prompts import messages, render
from finqa_ft.retrieval import Retriever


class PowerMeter(threading.Thread):
    """Samples GPU board power with nvidia-smi once per second."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[float] = []
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.wait(1.0):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            ).stdout.split()
            if out:
                self.samples.append(float(out[0]))


def weights_gb(model: str) -> float:
    path = Path(model)
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model, allow_patterns=["*.safetensors"]))
    return sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9


def build_prompts(args, tokenizer, exs) -> list[str]:
    retriever = Retriever(load_split("train")) if args.shots else None
    return [
        render(
            tokenizer,
            messages(e, retriever.top_k(e.question, args.shots) if retriever else ()),
        )
        for e in exs
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--limit", type=int, default=None, help="seeded random subset size")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--guided", action="store_true", help="constrain output to the DSL grammar")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--latency-n", type=int, default=50)
    ap.add_argument("--price-kwh", type=float, default=0.30, help="electricity price, EUR/kWh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    exs = load_split(args.split)
    if args.limit:
        exs = random.Random(args.seed).sample(exs, args.limit)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        seed=args.seed,
        enable_prefix_caching=False,
    )
    prompts = build_prompts(args, llm.get_tokenizer(), exs)
    params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
            structured_outputs=(
                StructuredOutputsParams(regex=program_regex(e.table)) if args.guided else None
            ),
        )
        for e in exs
    ]

    meter = PowerMeter()
    meter.start()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, params)
    batch_s = time.perf_counter() - t0
    meter.stop.set()

    rows = []
    for e, o in zip(exs, outs, strict=True):
        c = o.outputs[0]
        rows.append(
            score(e, c.text)
            | {
                "output": c.text,
                "prompt_tokens": len(o.prompt_token_ids),
                "output_tokens": len(c.token_ids),
                "hit_max_tokens": c.finish_reason == "length",
            }
        )
    n = args.latency_n
    for r, p, sp in zip(rows[:n], prompts[:n], params[:n], strict=True):
        t = time.perf_counter()
        llm.generate([p], sp, use_tqdm=False)
        r["latency_s"] = time.perf_counter() - t

    train_reports = {e.report for e in load_split("train")}
    summary = summarize(rows, {e.report for e in exs} - train_reports)
    watts = sum(meter.samples) / max(len(meter.samples), 1)
    sec_per_q = batch_s / len(exs)
    cfg = llm.llm_engine.model_config
    summary |= {
        "tag": args.tag,
        "model": args.model,
        "split": args.split,
        "limit": args.limit,
        "shots": args.shots,
        "guided": args.guided,
        "dtype": str(cfg.dtype),
        "quantization": cfg.quantization or "none",
        "weights_gb": round(weights_gb(args.model), 2),
        "vllm_gpu_mem_budget_gb": round(args.gpu_mem * 24, 1),
        "hit_max_tokens": sum(r["hit_max_tokens"] for r in rows),
        "prompt_tokens_mean": sum(r["prompt_tokens"] for r in rows) / len(rows),
        "batch_seconds": batch_s,
        "batch_seconds_per_question": sec_per_q,
        "throughput_q_per_s": 1 / sec_per_q,
        "mean_power_w": watts,
        "cost_eur_per_1000q": watts * sec_per_q * 1000 / 3.6e6 * args.price_kwh,
        "price_eur_per_kwh": args.price_kwh,
    }
    for sub in ("predictions", "metrics"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)
    name = f"{args.tag}_{args.split}" + (f"_n{args.limit}" if args.limit else "")
    pred_path = args.out_dir / "predictions" / f"{name}.jsonl"
    pred_path.write_text("\n".join(json.dumps(r) for r in rows))
    metrics_path = args.out_dir / "metrics" / f"{name}.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("tag", "n", "exec_acc", "valid_rate", "batch_seconds_per_question")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
