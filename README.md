# finqa-lora

[![ci](https://github.com/Tanatas298/finqa-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanatas298/finqa-lora/actions/workflows/ci.yml)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20model-qwen3--4b--finqa--lora-yellow)](https://huggingface.co/Tanatas298/qwen3-4b-finqa-lora)

A small open model that answers numerical questions about company annual reports.

Qwen3-4B is fine-tuned with LoRA on [FinQA](https://github.com/czyssrs/FinQA) to read a report
page (text and a table) and write a short calculation program. Python executes the program, so
the arithmetic is exact and every answer can be traced step by step.

## Highlights

- **75.7% accuracy** on the FinQA test set, averaged over two training runs: 75.3% for
  run 1 (seed 0) and 76.0% for run 2 (seed 1).
- **Beats a model 3.5× larger:** +7 points over prompted Qwen3-14B, and +5.7 points even when
  the 14B model is forced to produce valid programs.
- **2.7× faster:** 0.40 s per answer vs 1.07 s for the 14B model, in 8 GB of weights.
- **Cheap to train:** 6,251 examples, under 2 hours on one 24 GB GPU.
- **Generalises:** similar accuracy on the 103 test questions from company-years not in the
  training set (75.7% vs 75.3% overall for run 1; wide confidence interval, 67.0–83.5%).

## Example

<!-- BEGIN example -->
> what was the percentage change in net earnings from continuing operations from 2014 to 2015?

|  | 2015 | 2014 |
|---|---|---|
| net earnings from continuing operations | 3538 | 3480 |

```text
Program: subtract(3538, 3480), divide(#0, 3480)
#0 = subtract(3538, 3480) = 58
#1 = divide(#0, 3480) = 0.01667
```
<!-- END example -->

## Results

FinQA test set, 1,147 questions, scored with FinQA's official evaluation script. Prompted
models get 4 similar solved examples in the prompt. *Guided decoding* restricts the output to
valid programs. Ranges are 95% bootstrap confidence intervals.

<!-- BEGIN results -->
| Model | Accuracy (95% CI) | Program match | Valid output | Latency | Size |
|---|---|---|---|---|---|
| Qwen3-4B, prompted | 59.7% (56.8–62.5) | 54.8% | 89.4% | 0.59 s | 8 GB |
| Qwen3-4B, prompted + guided | 62.3% (59.5–65.1) | 56.7% | 99.0% | 0.52 s | 8 GB |
| Qwen3-8B, prompted | 53.9% (51.0–56.8) | 49.2% | 83.9% | 2.90 s | 16 GB |
| Qwen3-14B (4-bit), prompted | 68.3% (65.6–70.9) | 63.1% | 93.6% | 1.07 s | 10 GB |
| Qwen3-14B (4-bit), prompted + guided | 69.7% (67.0–72.3) | 64.3% | 99.5% | 0.98 s | 10 GB |
| **Qwen3-4B + LoRA, run 1 (seed 0)** | **75.3% (72.8–77.9)** | **71.0%** | **99.6%** | **0.40 s** | **8 GB** |
| **Qwen3-4B + LoRA, run 2 (seed 1)** | **76.0% (73.5–78.5)** | **71.6%** | **99.9%** | **0.41 s** | **8 GB** |
| **Qwen3-4B + LoRA, mean of the 2 runs** | **75.7% (73.3–78.0)** | **71.3%** | **99.7%** | **0.41 s** | **8 GB** |
<!-- END results -->

- **Accuracy:** the executed program gives the correct answer.
- **Program match:** the program is equivalent to the reference program.
- **Valid output:** the output is a program that runs.

The fine-tuned model's lead over every baseline is statistically significant (p < 0.0001,
[details](docs/results.md#significance)).

### Two training runs

The final configuration was trained twice: run 1 (seed 0) and run 2 (seed 1).

<!-- BEGIN runs -->
|  | Run 1 (seed 0) | Run 2 (seed 1) | Mean |
|---|---|---|---|
| Accuracy | 75.3% | 76.0% | 75.7% |
| Program match | 71.0% | 71.6% | 71.3% |
| Accuracy on reports unseen in training | 75.7% | 74.8% | 75.2% |
<!-- END runs -->

The model on the Hugging Face Hub is run 1 (seed 0), chosen because it scored higher on the
validation set (76.6% vs 75.9%).

### Training data size

<!-- BEGIN curve -->
| Training examples | 500 | 1,500 | 6,251 |
|---|---|---|---|
| Accuracy | 68.6% | 71.4% | 75.3% |
<!-- END curve -->

With 500 examples the model already matches prompted Qwen3-14B.

<p align="center">
  <img src="reports/acc_vs_size.png" width="49%" alt="Accuracy vs model size">
  <img src="reports/data_efficiency.png" width="49%" alt="Accuracy vs training examples">
</p>

### Published results (for context)

| System | Model | Accuracy |
|---|---|---|
| FinQANet (Chen et al., 2021) | fine-tuned RoBERTa-large | 61.2% |
| APOLLO (Sun et al., 2024) | fine-tuned ensemble | 71.1% |
| FINDER (Khatuya et al., 2025) | GPT-4 pipeline | 75.3% |
| **This project** | **Qwen3-4B + LoRA** | **75.3–76.0%** |

Numbers as reported in each paper; the setups are not identical. The published systems first
retrieve a few facts from the page, while this project gives the model the whole page. FINDER
writes Python programs instead of FinQA programs, and its exact scoring rule is not stated in
the paper. GPT-4 with chain-of-thought reaches 78.0% with free-form answers and a different
scoring method (Li et al., 2023), so it is not listed in the table. Details:
[`docs/evaluation.md`](docs/evaluation.md#published-systems).

## How it works

1. **Data:** each FinQA question comes with a report page and a gold program such as
   `subtract(3538, 3480), divide(#0, 3480)`. The model learns to write that program.
2. **Training:** LoRA (rank 16) on Qwen3-4B in bf16, 2 epochs, loss on the program only.
   Learning rate and number of epochs were chosen on the validation set.
3. **Inference:** vLLM generates the program; a Python interpreter executes it.
4. **Evaluation:** FinQA's official script compares the result with the gold answer. The test
   set was used only for the final numbers.

More detail: [`docs/results.md`](docs/results.md) (all results, confidence intervals, error
analysis), [`docs/evaluation.md`](docs/evaluation.md) (how everything is measured) and
[`docs/decisions/`](docs/decisions/) (why the main design choices were made).

## Limitations

- **One dataset:** FinQA only, English-language annual reports of US companies (1999–2019).
- **Possible contamination:** FinQA has been public since 2021 and may be in Qwen3's
  pre-training data. This would affect the baselines and the fine-tuned model alike.
- **Two training runs:** enough to show the result is stable, not to estimate the spread.
- **Baselines:** Qwen3-14B runs in 4-bit and without thinking mode; a larger or reasoning
  model was not tested.
- **Setup differs from published work:** the model reads the whole page instead of retrieved
  facts, so comparisons with papers are indicative.
- **Noisy labels:** an estimated 7–12% of test questions have a wrong or ambiguous gold
  answer, which caps every model's score.

## Quick start

Use the trained model from the Hub:
[`Tanatas298/qwen3-4b-finqa-lora`](https://huggingface.co/Tanatas298/qwen3-4b-finqa-lora)
(usage example in the model card).

Reproduce everything (Linux or WSL2; vLLM and Unsloth need separate environments). The
`requirements*.txt` files pin the direct dependencies; `requirements-vllm.lock` and
`requirements-train.lock` pin the full environments that produced the results.

```bash
python3 -m venv ~/venvs/vllm && ~/venvs/vllm/bin/pip install -r requirements.txt
python3 -m venv ~/venvs/train && ~/venvs/train/bin/pip install -r requirements-train.txt
export PYTHONPATH=src

~/venvs/vllm/bin/python scripts/prepare_data.py --tokenizer Qwen/Qwen3-4B  # download + checks
bash scripts/run_all.sh                            # train, evaluate, build tables
```

Rebuild all tables and charts from the stored predictions (no GPU; the first command
downloads FinQA, which the analysis needs):

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python scripts/prepare_data.py --download-only
PYTHONPATH=src python scripts/make_tables.py
```

## Project structure

```text
src/finqa_ft/     interpreter, prompts, metrics, official FinQA scorer, guided-decoding grammar
scripts/          data preparation, training, evaluation, tables and charts, Hub upload
reports/          scored predictions, metrics, tables and charts
docs/             full results, evaluation method, design decisions, model card
tests/            unit tests (no GPU needed): pytest && ruff check .
```

## License

Code: MIT. FinQA dataset: Chen et al., EMNLP 2021 (MIT / CC BY 4.0), not redistributed here.
Qwen3: Apache-2.0. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
