# 004: Baselines

**Decision:** compare against the strongest local prompting we can run on the same GPU, on the
same questions, through the same vLLM code path (`scripts/evaluate.py`).

| Baseline | Why |
|---|---|
| Qwen3 4B zero-shot | the student before training |
| Qwen3 4B, 4 retrieved shots | the strong prompting baseline for the same model |
| Qwen3 8B, 4 shots | next size up |
| Qwen3 14B AWQ, 4 shots | bf16 14B (about 28 GB) does not fit 24 GB, so the official AWQ 4-bit |

- Shots are retrieved from train by TF-IDF question similarity and shown as question,
  gold supporting facts and program (full contexts would not fit 4 shots in the window).
- No paid APIs: the "large model" reference is the largest Qwen3 run locally.
- Dropped by decision on 2026-10-31 to save GPU time: Qwen3 14B with thinking mode (estimated
  1-1.5 h on test) and Qwen3 32B AWQ (about 1 h plus a 19 GB download). The strongest
  reference is therefore 14B AWQ few-shot without thinking; a thinking model might close more
  of the gap, and this is not measured.
- Greedy decoding, seed 0, thinking off everywhere.
- Added 2026-11-03: the 4B and 14B few-shot baselines are also run with guided decoding
  (`--guided`, grammar in `src/finqa_ft/grammar.py`), so they cannot emit an invalid program.
  This separates format errors from reasoning errors.
- The 8B row is kept as run. It writes a prose explanation before the program in 82% of
  answers; thinking mode was off and no `<think>` block appears. A rerun would give identical
  greedy outputs, so it was not repeated.
