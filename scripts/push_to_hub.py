"""Upload the LoRA adapter and model card to the Hugging Face Hub.

Log in first with `hf auth login` (the token is stored by huggingface_hub, never in this repo).
Usage: python scripts/push_to_hub.py outputs/qwen3-4b-lora-full/adapter
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Tanatas298/qwen3-4b-finqa-lora"
FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", type=Path)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--card", type=Path, default=Path("docs/model_card.md"))
    args = ap.parse_args()
    missing = [f for f in FILES if not (args.adapter / f).exists()]
    assert not missing, f"missing in {args.adapter}: {missing}"

    api = HfApi()
    print("logged in as", api.whoami()["name"])
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=args.adapter,
        allow_patterns=list(FILES),
        commit_message="Upload LoRA adapter (seed 0, chosen on validation)",
    )
    api.upload_file(
        repo_id=args.repo,
        path_or_fileobj=args.card,
        path_in_repo="README.md",
        commit_message="Model card",
    )
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
