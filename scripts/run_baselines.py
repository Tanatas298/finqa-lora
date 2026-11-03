import argparse
import subprocess
import sys

SHOTS = "4"
BASELINES = {
    "qwen3-4b-zeroshot": ["--model", "Qwen/Qwen3-4B"],
    "qwen3-4b-fewshot": ["--model", "Qwen/Qwen3-4B", "--shots", SHOTS],
    "qwen3-8b-fewshot": ["--model", "Qwen/Qwen3-8B", "--shots", SHOTS],
    "qwen3-14b-awq-fewshot": ["--model", "Qwen/Qwen3-14B-AWQ", "--shots", SHOTS],
    "qwen3-4b-fewshot-guided": ["--model", "Qwen/Qwen3-4B", "--shots", SHOTS, "--guided"],
    "qwen3-14b-awq-fewshot-guided": ["--model", "Qwen/Qwen3-14B-AWQ", "--shots", SHOTS, "--guided"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", default=None)
    ap.add_argument("--only", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    args = ap.parse_args()
    failed = []
    for tag in args.only:
        cmd = [sys.executable, "scripts/evaluate.py", "--tag", tag, "--split", args.split]
        cmd += BASELINES[tag] + (["--limit", args.limit] if args.limit else [])
        print(">>", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode:
            failed.append(tag)
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
