import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter/checkpoint into bf16 weights")
    ap.add_argument("adapter", help="adapter or checkpoint directory (has adapter_config.json)")
    ap.add_argument("out")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        args.adapter, max_seq_length=args.max_seq_len, load_in_4bit=False, use_exact_model_name=True
    )
    tmp = Path(args.out + ".tmp")  # renamed when complete, so a killed merge never looks done
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained_merged(str(tmp), tok, save_method="merged_16bit")
    tmp.rename(args.out)


if __name__ == "__main__":
    main()
