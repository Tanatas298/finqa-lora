---
base_model: Qwen/Qwen3-4B
library_name: peft
license: apache-2.0
language:
- en
datasets:
- ibm-research/finqa
pipeline_tag: text-generation
tags:
- finance
- finqa
- numerical-reasoning
- lora
- peft
- program-synthesis
model-index:
- name: qwen3-4b-finqa-lora
  results:
  - task:
      type: question-answering
      name: Numerical reasoning over financial reports
    dataset:
      name: FinQA (test, 1,147 questions)
      type: ibm-research/finqa
      split: test
    metrics:
    - type: accuracy
      name: Execution accuracy (official FinQA script)
      value: 75.3
    - type: accuracy
      name: Program accuracy (official FinQA script)
      value: 71.0
---

# qwen3-4b-finqa-lora

A LoRA adapter for [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) that answers numerical
questions about a financial report page by writing a program in FinQA's DSL, for example
`subtract(3538, 3480), divide(#0, 3480)`. The program is executed in Python, so the model never
does the arithmetic itself. Code, evaluation and all results:
[github.com/Tanatas298/finqa-lora](https://github.com/Tanatas298/finqa-lora).

## Results (FinQA test, 1,147 questions, official FinQA evaluation script)

| Model | Execution accuracy [95% CI] | Program accuracy |
|---|---|---|
| **This adapter: run 1 (seed 0)** | **75.3 [72.8, 77.9]** | 71.0 |
| Same recipe, run 2 (seed 1), not published | 76.0 [73.5, 78.5] | 71.6 |
| Qwen3-14B AWQ, 4 retrieved examples, guided decoding | 69.7 [67.0, 72.3] | 64.3 |
| Qwen3-14B AWQ, 4 retrieved examples | 68.3 [65.6, 70.9] | 63.1 |
| Qwen3-4B, 4 retrieved examples | 59.7 [56.8, 62.5] | 54.8 |

95% CIs from a paired bootstrap (10,000 resamples). For reference, FINDER (FLAN-T5 retriever +
GPT-4, EMNLP 2025) reports 75.32 and the APOLLO ensemble (LREC-COLING 2024) 71.07; see the
repository for what is and is not directly comparable.

**Which run this is.** The recipe was trained twice: run 1 (seed 0) and run 2 (seed 1). Run 1
is published because it scored higher on the validation split (76.6% vs 75.9% official execution
accuracy). Run 2 is higher on test (76.0% vs 75.3%); test results played no part in the
choice.

## Training

- Base model: Qwen/Qwen3-4B, bf16, chat template with thinking disabled.
- Data: the 6,251 FinQA training questions. Input: the report page (text before the table,
  the table in Markdown, text after the table) and the question. Target: `Program: <gold
  program>`. Loss on the target only.
- LoRA r=16, alpha=32, dropout 0, on q, k, v, o, gate, up and down projections.
- 2 epochs (782 steps), learning rate 2e-4, cosine schedule, warm-up 3%, effective batch 16,
  max length 4,096. Learning rate and epoch count chosen on validation.
- Hardware: one 24 GB GPU, Unsloth + TRL. Run 2 (seed 1) ran without interruption: 1.8 hours,
  peak 9.5 GB of GPU memory. Run 1 (seed 0, this adapter) was resumed after an interruption,
  so its total time was not recorded; peak 9.6 GB.

## Usage

The adapter expects the exact prompt used in training, built by `finqa_ft.prompts.messages`
in the repository (system prompt describing the DSL, then the page and the question).

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finqa_ft.data import Example  # pip install git+https://github.com/Tanatas298/finqa-lora
from finqa_ft.program import execute
from finqa_ft.prompts import extract_program, messages, render

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", torch_dtype="bfloat16", device_map="auto"
)
model = PeftModel.from_pretrained(model, "Tanatas298/qwen3-4b-finqa-lora")

table = [["", "2015", "2014"], ["net earnings from continuing operations", "3538", "3480"]]
ex = Example(
    id="demo",
    filename="demo",
    question="what was the percentage change in net earnings from 2014 to 2015?",
    pre_text=[],
    post_text=[],
    table=table,
    program="",
    answer="",
    gold_inds={},
)
inputs = tok(render(tok, messages(ex)), return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
program = extract_program(tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True))
print(program, "=", execute(program, table))  # subtract(3538, 3480), divide(#0, 3480) = 0.01666...
```

For serving, merge the adapter and use vLLM (`scripts/serve.sh` in the repository).

## Intended use and limitations

- Intended for research and education on numerical reasoning over financial tables and text,
  and as a reproducible baseline for FinQA-style tasks.
- Trained and evaluated on one dataset: English-language annual reports of S&P 500 companies,
  1999–2019. Accuracy on other documents, languages or question styles is unknown.
- About one answer in four on the FinQA test set is wrong. Part of FinQA's gold labels are
  themselves wrong or ambiguous (an estimated 7–12% of test questions).
- FinQA has been public since 2021 and may be part of Qwen3's pre-training data.
- Two training runs only; the published run 1 (seed 0) is not the better one on test.
- **Not financial advice.** Outputs must not be used for investment, accounting, audit or any
  other financial decision without independent verification.

## Licence and credit

- Adapter: Apache-2.0, following the base model
  [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) (Apache-2.0, Qwen team).
- Training data: FinQA, Chen et al., *FinQA: A Dataset of Numerical Reasoning over Financial
  Data*, EMNLP 2021 ([GitHub](https://github.com/czyssrs/FinQA), MIT;
  [Hugging Face](https://huggingface.co/datasets/ibm-research/finqa), CC BY 4.0). The
  underlying reports are public SEC filings.

```bibtex
@inproceedings{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana
          and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao
          and Routledge, Bryan R and Wang, William Yang},
  booktitle={EMNLP},
  year={2021}
}
```
