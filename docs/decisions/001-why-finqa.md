# 001: Why FinQA

**Decision:** fine-tune on FinQA (Chen et al., 2021), official splits.

- `ticket-triage-lora` failed because its labels were noisy and a TF-IDF classifier won. FinQA
  answers are numbers with a gold program, so correctness is checked automatically and labels
  are consistent.
- The model must read a table plus text and calculate. Bag-of-words classifiers cannot do this,
  so the cheap baseline has to be a prompted LLM.
- Published work reports large gains of fine-tuned small models over their zero/few-shot
  versions on FinQA, so fine-tuning has a realistic chance to earn its place.

**Caveat found in Step 0:** the official splits are disjoint by report *page* (0 shared pages),
not by company-year filing (197 company-years shared between train and validation, 243 between
train and test). We keep the official splits for comparability and additionally report test
accuracy on questions from company-years never seen in training.
