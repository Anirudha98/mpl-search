# mpl-search

Find the matplotlib function for a question asked in everyday words.

People don't ask "how do I use `minorticks_on`". They ask "add minor gridlines", "remove the
top and right axis", "make the bars show their values". The words in the question and the
words in the documentation rarely match, so keyword search fails exactly where help is needed.

This project measures that gap on real Stack Overflow questions, narrows it by fine-tuning a
small embedding model, and answers questions with an LLM that must cite the documentation it
used, in structured JSON.

## Results

Test split: 568 Stack Overflow questions never used for training or model choice. 492 are
**hard**: none of the functions the answer used is named in the title. R@5 = share of questions
with a correct function in the top 5 results.

| Method | R@5 | R@5 hard | MRR@10 |
|---|---|---|---|
| BM25 keyword search | 0.16 | 0.10 | 0.11 |
| all-MiniLM-L6-v2, off the shelf | 0.18 | 0.13 | 0.13 |
| bge-base-en-v1.5, off the shelf (4x larger) | **0.24** | 0.16 | **0.16** |
| **MiniLM fine-tuned on Stack Overflow pairs** | 0.22 | **0.19** | 0.14 |

Fine-tuned vs the same model off the shelf, 95% paired bootstrap interval over questions:
**+0.06 on hard questions [+0.02, +0.10]**, +0.04 overall [+0.00, +0.08]. Fine-tuning helps
where the wording differs, which is the point; a 4x larger model off the shelf is better on
easy questions. The model was picked on the validation split before test was scored
(`data/test_results_auto_labels.txt`).

### Why every method stops near 0.2

Reading the misses (EXPERIMENTS.md, section 6), the ceiling is mostly the labels, not the
models. A label is the set of functions one accepted answer happened to call:

- "Add minor gridlines": the model's top hit is `minorticks_on`, a correct answer. The label
  only has `set_minor_locator`, the route that answer took.
- "Modify legend color …" is labelled `tight_layout`, an incidental call in the answer.

So automatic scores undercount every method. `label.py` builds a method-blind, hand-checked
sample of 100 test questions where any correct function counts.

| On 100 hand-checked test questions | R@5 |
|---|---|
| *pending: run `label.py`, then `evaluate.py --gold`* | |

### Cited answers

`answer.py` gives the top 3 functions to `meta/llama-3.3-70b-instruct` (NVIDIA NIM), which must
reply with `{"found", "answer", "citations"}` or say it could not find the answer. Replies that
are not JSON, or that cite an excerpt it was not shown, are counted as failures.

*Pending: results on 100 test questions.*

## How it works

| Step | File | What it does |
|---|---|---|
| Corpus | `corpus.py` | 738 documents, one per public function or method name in pyplot, Axes, Figure, Axis, Legend, Text, Line2D, Patch, Colorbar, colors, ticker, dates. Same-named methods are merged: answers call `plt.legend` and `ax.legend` interchangeably |
| Data | `collect.py` | 4,964 Stack Overflow questions with an accepted answer. Label = matplotlib functions the answer's code calls that the question's code did not. Scaffolding (`subplots`, `figure`, `plot`, `show`) and names that collide with Python (`dict.get`) are dropped. 2,688 questions keep a label; split 70/10/20 by question id |
| Evaluation | `evaluate.py` | BM25 (written here, ~20 lines), dense retrieval, R@1, R@5, MRR@10, a hard subset, paired bootstrap intervals |
| Fine-tuning | `finetune.py` | (title, documentation) pairs from the train split only; in-batch negatives with no duplicate documents per batch; lr 1e-4, 5 epochs, about 6 minutes on an Apple M5 |
| Hand-checking | `label.py` | Pools every method's top 5 plus the automatic labels, shuffled, so the judge can't tell which method proposed what |
| Answers | `answer.py` | Retrieval + LLM with JSON output and citation checks |

Every script has a `--check` self-test that needs no network and no model.

## Run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "matplotlib==3.10.*" "sentence-transformers==5.*" torch numpy requests datasets accelerate

.venv/bin/python corpus.py
.venv/bin/python collect.py          # Stack Exchange API, no key needed; cached in data/cache/
.venv/bin/python evaluate.py --split val
.venv/bin/python finetune.py         # -> models/final
.venv/bin/python evaluate.py --split test --model models/final
.venv/bin/python label.py --pool 100 && .venv/bin/python label.py && .venv/bin/python evaluate.py --gold
echo "NVIDIA_API_KEY=..." > .env && .venv/bin/python answer.py "how do I move the legend outside the plot"
```

`data/corpus.jsonl` and `data/questions.jsonl` are committed, so the collection step can be skipped.

## Limits

- Labels come from one accepted answer per question, so they miss other valid solutions
  and include incidental calls (see above).
- Queries are titles only. A real user might type more, or less.
- One training run per setting. Validation has 260 questions, so differences under about
  0.06 R@5 are noise; that is why every comparison above has an interval.
- The fine-tuned model still memorises: R@5 0.70 on its own training questions.
- The corpus covers the main matplotlib classes, not every submodule (e.g. no `mpl_toolkits`).

## Data

Questions and answers are from Stack Overflow, licensed CC BY-SA 4.0; every row in
`data/questions.jsonl` keeps its link. Documentation text is from matplotlib's docstrings
(matplotlib license).
