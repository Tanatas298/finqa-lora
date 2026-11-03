import argparse
import json
import random
import shutil
import time
from pathlib import Path

import yaml

from finqa_ft.data import load_split
from finqa_ft.prompts import messages, render, target


def steady(marks: list[tuple[float, int]], skip: int = 5) -> dict:
    """Tokens/s and s/step after the first optimizer steps (excludes compile and warm-up)."""
    if len(marks) <= skip:
        return {}
    (t0, n0), (t1, n1) = marks[skip - 1], marks[-1]
    return {
        "steady_tokens_per_second": (n1 - n0) / (t1 - t0),
        "steady_seconds_per_step": (t1 - t0) / (len(marks) - skip),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train_4b.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. lora_r=32 epochs=1")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    from unsloth import FastLanguageModel  # must load before transformers and trl

    # isort: split
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    marks: list[tuple[float, int]] = []

    class StepTimer(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            marks.append((time.perf_counter(), state.num_input_tokens_seen))

    model, tok = (
        FastLanguageModel.from_pretrained(  # exact repo: same weights/template as baselines
            cfg["model"],
            max_seq_length=cfg["max_seq_len"],
            dtype=torch.bfloat16,
            load_in_4bit=False,
            use_exact_model_name=True,
        )
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    exs = load_split("train")
    random.Random(cfg["seed"]).shuffle(exs)
    exs = exs[: cfg["train_limit"]]  # prefixes of one shuffle: the 500 subset is inside the 1500
    rows = [{"prompt": render(tok, messages(e)), "completion": target(e)} for e in exs]
    lens = [len(tok(r["prompt"] + r["completion"]).input_ids) for r in rows]
    kept = [r for r, n in zip(rows, lens, strict=True) if n <= cfg["max_seq_len"]]
    dropped = len(rows) - len(kept)
    print(f"training on {len(kept)} examples, {dropped} longer than max_seq_len dropped")

    out = Path(cfg["output_dir"])
    timing = cfg.get("max_steps", -1) > 0

    class EpochSnapshot(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            snap = out / "checkpoints" / f"checkpoint-{state.global_step}"
            model.save_pretrained(snap)
            tok.save_pretrained(snap)

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=Dataset.from_list(kept),
        args=SFTConfig(
            output_dir=str(out / "resume"),
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            num_train_epochs=cfg["epochs"],
            max_steps=cfg.get("max_steps", -1),
            learning_rate=cfg["learning_rate"],
            warmup_ratio=cfg["warmup_ratio"],
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=10,
            save_strategy="no" if timing else "steps",
            save_steps=cfg.get("save_steps", 100),
            save_total_limit=1,
            include_num_input_tokens_seen="non_padding",
            seed=cfg["seed"],
            max_length=cfg["max_seq_len"],
            completion_only_loss=True,
            report_to="none",
        ),
        callbacks=[StepTimer()] + ([] if timing else [EpochSnapshot()]),
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    resume = get_last_checkpoint(str(out / "resume")) if (out / "resume").is_dir() else None
    result = trainer.train(resume_from_checkpoint=resume)
    wall = time.perf_counter() - t0

    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    if not args.no_merge:
        tmp = out / "merged.tmp"  # renamed when complete, so a killed merge never looks done
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained_merged(str(tmp), tok, save_method="merged_16bit")
        tmp.rename(out / "merged")
    run = cfg | {
        "examples": len(kept),
        "dropped_too_long": dropped,
        "resumed_from": resume,
        "steps": trainer.state.global_step,
        "wall_seconds": wall,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "train_tokens_per_second": trainer.state.num_input_tokens_seen / wall,
        **steady(marks),
        "final_train_loss": result.training_loss,
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/train_runs.jsonl", "a") as f:
        f.write(json.dumps(run) + "\n")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
