# Third-party notices

## FinQA dataset

Chen et al., *FinQA: A Dataset of Numerical Reasoning over Financial Data*, EMNLP 2021.
https://arxiv.org/abs/2109.00122

- Source used: original release, https://github.com/czyssrs/FinQA (`dataset/train.json`,
  `dev.json`, `test.json`), downloaded by `scripts/prepare_data.py`. Not redistributed here.
- Licence: the GitHub repository is MIT-licensed; the Hugging Face card
  (`ibm-research/finqa`, which wraps the same GitHub archive) declares CC BY 4.0. We credit the
  authors under both.
- The underlying reports are public SEC filings of S&P 500 companies.

```
@inproceedings{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana
          and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao
          and Routledge, Bryan R and Wang, William Yang},
  booktitle={EMNLP},
  year={2021}
}
```

## FinQA evaluation script

`src/finqa_ft/vendor/finqa_evaluate.py` is an unmodified copy of `code/evaluate/evaluate.py` from
https://github.com/czyssrs/FinQA (commit `dfc5b72c01`), MIT licence, Copyright (c) 2021
Zhiyu Chen. It is used, through `src/finqa_ft/official.py`, for the headline metrics.

## Models

Qwen3 models (Qwen/Qwen3-4B, Qwen/Qwen3-8B, Qwen/Qwen3-14B-AWQ) by the Qwen team, Apache-2.0.
Fine-tuned weights produced by this project inherit Apache-2.0 and are not committed.

## Software

vLLM (Apache-2.0), Unsloth (Apache-2.0), Hugging Face Transformers, TRL, PEFT, Datasets
(Apache-2.0), matplotlib (PSF-based licence).
