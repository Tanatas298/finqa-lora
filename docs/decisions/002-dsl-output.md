# 002: Programs in the FinQA DSL instead of free-text answers

**Decision:** every model outputs a FinQA program (e.g. `subtract(5829, 5735), divide(#0, 5735)`),
which `src/finqa_ft/program.py` executes.

- Arithmetic is done by Python, not by the model. Errors are reading/planning errors, which is
  what we want to measure.
- The gold program is the supervision target for LoRA, so training is short and consistent.
- Answers are checked by execution. Headline numbers use FinQA's official script (exact match
  after rounding to 5 decimals); the project's original lenient metric (0.1% relative, plus a
  factor-100 percent/fraction allowance) is kept as a secondary column. See
  `docs/evaluation.md`.
- Programs are inspectable: the demo prints each step.
- Prompted baselines get the same DSL description and must end their answer with
  `Program: ...`.

Interpreter check: gold programs reproduce the gold answers on 100% of train, validation and test (strict
tolerance) after handling FinQA's duplicated cell values (`-13 ( 13 )`) and duplicate row
names (gold uses the last matching row).
