# 003: LoRA and its settings

**Decision:** bf16 LoRA on Qwen3 4B with Unsloth, all attention and MLP projections, gradient
checkpointing, loss on the program only (prompt masked).

- A 4B model in bf16 (about 8 GB) plus LoRA and activations fits a 24 GB RTX 3090; full
  fine-tuning does not.
- bf16 rather than QLoRA: the model fits, and 4-bit base weights would add a quality confound.
- Fixed: r=16, alpha=32, effective batch 16, cosine schedule. Unsloth batches padding-free
  (examples concatenated, attention kept per example via sequence lengths), so no padding is
  wasted. Checked: loss falls only on the `Program: ...<|im_end|>` tokens (85 of 6,221 tokens
  in one batch).
- Search (validation only, 2 runs): 1,500 examples, 1 epoch, lr 1e-4 vs 2e-4. The better run
  doubles as the 1,500 point of the data-efficiency curve; a 500-example run is added with the
  same settings. Subsets are prefixes of one seeded shuffle, so 500 is inside 1,500.
- Final: all 6,251 examples, 2 epochs, checkpoint after each epoch; both are evaluated on
  validation and the better one goes to test. The epoch-1 checkpoint also gives a consistent
  1-epoch data-efficiency curve (500 / 1,500 / 6,251).
- Time estimates come from a 30-step timing run (steady tokens/s after warm-up).
  All runs are logged in `reports/train_runs.jsonl`.
- Training uses the same chat-rendered prompt as evaluation (`prompts.render`, thinking off),
  so train and test inputs are identical.
- Examples longer than `max_seq_len` (chosen to cover at least 99%) are dropped from training
  and counted; evaluation never truncates.
