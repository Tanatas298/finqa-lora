# Evaluation

How the numbers in [results.md](results.md) are produced. Results live there; this page only
describes the setup.

## Task and input

Each FinQA question comes with one page of an annual report: the text before the table, the
table and the text after it. Every model sees that whole page (the table as Markdown) and the
question, and must answer with a program in FinQA's DSL:

```text
Program: subtract(3538, 3480), divide(#0, 3480)
```

The program is executed by `src/finqa_ft/program.py`. The models never see the gold supporting
facts of the question they answer.

## Models compared

| Model | Setup |
|---|---|
| Qwen3-4B, zero-shot | DSL description only |
| Qwen3-4B / 8B / 14B, prompted | plus 4 solved training examples, retrieved by question similarity (TF-IDF) |
| Qwen3-4B / 14B, prompted + guided | same prompt, decoding restricted to valid programs |
| Qwen3-4B + LoRA | fine-tuned, no examples in the prompt |

- All models run through the same vLLM code (`scripts/evaluate.py`): greedy decoding, thinking
  mode off, up to 256 output tokens.
- Qwen3-14B is the official AWQ 4-bit checkpoint, because the 16-bit model does not fit in
  24 GB. The other models run in bf16.
- **Guided decoding** (`src/finqa_ft/grammar.py`) allows only `Program: ` followed by valid
  operations, numbers, constants, references to earlier steps and the row names of the
  question's own table. It separates format errors from reasoning errors.
- **Qwen3-8B** writes a prose explanation before the program in 82% of its answers (4B and 14B:
  under 4%). This makes it slower and less accurate than 4B in this setup. No reasoning
  (`<think>`) block appears in any output.

## Metric

**Accuracy** is FinQA's official execution accuracy: the program's result, rounded to 5
decimals, must equal the gold answer exactly. **Program match** is FinQA's official program
accuracy: the program must be symbolically equivalent to the gold program. Both come from
FinQA's own evaluation script (`src/finqa_ft/vendor/finqa_evaluate.py`, copied unchanged from
the [FinQA repository](https://github.com/czyssrs/FinQA), commit `dfc5b72c01`). The adapter in
`src/finqa_ft/official.py` reproduces the script's command-line output exactly on every
prediction file.

The project first used a **lenient** score: 0.1% tolerance, and answers off by exactly ×100
(percent vs fraction) count as correct. It was fixed before any test run and is still reported
as a secondary column. Differences from the official script:

| | Lenient (project) | Official |
|---|---|---|
| Number match | within 0.1% | exact after rounding to 5 decimals |
| Off by ×100 | correct | wrong |
| Table row names | case-insensitive | exact |
| Cells like `( 12 )` | read as −12 | invalid |
| Program match | normalised text, `add`/`multiply` order-free | symbolic equivalence |

Gold programs reproduce the gold answers on all 1,147 test questions under both.

**Output parsing.** The program is taken from the last `Program:` line, or the last line that
parses as a program. An audit of raw outputs found one parser bug (a program on the line after
a bare `Program:`). It was fixed and changed one output and no score. The other invalid outputs
are genuine model errors, such as nested calls (`multiply(divide(a, b), 100)`), unknown
operations, quoted row names or outputs cut off at the token limit.

## Statistics

- **Confidence intervals:** paired bootstrap over questions, 10,000 resamples, fixed seed. All
  models are resampled on the same questions.
- **Significance:** exact McNemar test on the questions where exactly one of two models is
  right.
- **Seeds:** the final configuration was trained twice. With two runs both values are shown and
  no standard deviation is claimed.
- Code: `src/finqa_ft/stats.py`, tested in `tests/test_stats.py`.

## Data hygiene

- **Validation for choices, test once.** Learning rate, number of epochs and the run published
  on the Hub were chosen on the validation set. The test set was used only to report final
  numbers.
- **Overlap.** FinQA's official splits share no report pages, but 243 annual reports (company
  and year) appear in both train and test. Results are also reported on the 103 test questions
  from reports not in the training set. Only 3 test questions come from a company absent from
  training, too few to score.
- **Contamination.** FinQA has been public since 2021, so Qwen3 may have seen it during
  pre-training. That would affect every model in the comparison, since all start from Qwen3
  weights. It cannot be measured here.

## Error labelling

100 failures of the fine-tuned model (run 1) were sampled with a fixed seed
(`scripts/error_sample.py`) and labelled by hand into eight categories
([error_labels.csv](../reports/error_labels.csv)). A gold label is counted as wrong or
ambiguous only when the model's own answer is a defensible reading of the question. Labels are
one annotator's judgement.

## Published systems

FinQA test accuracy as reported by each paper:

| System | Paper | Approach | Accuracy | Comparable? |
|---|---|---|---|---|
| Human expert | Chen et al., EMNLP 2021 | — | 91.2% | upper reference |
| GPT-4, chain-of-thought | Li et al., EMNLP 2023 | free-form answer | 78.0% | no: free-form answers, own scoring |
| FINDER | Khatuya et al., EMNLP 2025 | GPT-4 writes Python on retrieved facts | 75.3% | partly: Python programs |
| APOLLO, ensemble | Sun et al., LREC-COLING 2024 | fine-tuned retriever + generator, FinQA DSL | 71.1% | yes |
| FinQANet | Chen et al., EMNLP 2021 | fine-tuned RoBERTa-large, FinQA DSL | 61.2% | yes |

*Comparable* means the same task format (FinQA DSL programs read from the same page) and the
official script. The published systems first retrieve a few facts from the page; this project
gives the whole page to the model. Papers report a single run without confidence intervals.
Sources: FinQA (arXiv 2109.00122, Table 2), APOLLO (arXiv 2212.07249, Table 1), Li et al.
(arXiv 2305.05862, Table 8), FINDER (arXiv 2510.13157, Table 2).
