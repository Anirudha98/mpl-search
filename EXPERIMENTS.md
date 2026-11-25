# Experiments

Every run, including the ones that did not help. Scores are on the **validation** split
(162 questions, 134 hard) unless marked; the test split is only used for the final table
in the README. R@5 = share of questions with a correct function in the top 5.

With 162 questions, one question is 0.6 points, and differences under about 0.06 are
inside the bootstrap noise.

## 1. Baselines

| Method | R@5 | R@5 hard |
|---|---|---|
| BM25 on docstrings | 0.17 | 0.08 |
| all-MiniLM-L6-v2, off the shelf | 0.22 | 0.14 |

## 2. Fine-tuning, first try: lr 2e-5, 3 epochs

R@5 0.19 on validation, *below* the model it started from. On the training split itself it
reached only 0.26, and the loss ended at 2.8 against 3.47 for random guessing in a batch of 32:
the model was underfitting, not overfitting. The learning rate was too low for 2,952 pairs.

## 3. lr 1e-4, 5 epochs

| Split | R@5 | R@5 hard |
|---|---|---|
| train | 0.73 | 0.70 |
| validation | 0.27 | 0.21 |

Learns now, but the gap between train and validation says it memorises the 1,167 training
titles. Next step: more training questions.

## 4. More training data (1,167 -> 1,860 questions), lr 1e-4

The data collection added a third listing, which also grew validation from 162 to 260
questions. On the larger validation set:

| Model | R@5 | R@5 hard |
|---|---|---|
| BM25 | 0.18 | 0.10 |
| MiniLM off the shelf | 0.20 | 0.14 |
| fine-tuned, 5 epochs | 0.22 | 0.18 |
| fine-tuned, 3 epochs | 0.21 | 0.17 |

The 0.27 from run 3 did not hold up: it was the small validation set, not the model.
Fine-tuning still helps most on hard questions (+0.08 over BM25, interval +0.02 to +0.14).
Train R@5 is 0.70, so it still memorises.

## 5. Stronger base models, no training

| Model | R@5 | R@5 hard |
|---|---|---|
| bge-small-en-v1.5 | 0.21 | 0.15 |
| bge-base-en-v1.5 (4x larger than MiniLM) | 0.24 | 0.16 |

bge-base is best overall but not on hard questions. Against MiniLM off the shelf no
model's gain clears the noise on validation.

## 6. Why everything stops around 0.2: the labels

Every method, trained or not, lands between 0.18 and 0.24. Reading 15 validation misses of
the fine-tuned model:

- Often the model is right and the label is narrow. "Add minor gridlines": top result
  `minorticks_on`, a correct answer; the label only has `AutoMinorLocator`/`set_minor_locator`,
  the route that one accepted answer took. "Shared x-label and y-label": `supylabel` retrieved,
  label is `text`.
- Some labels are incidental calls in the answer, not the fix: "Modify legend color" is
  labelled `tight_layout`; "Bezier surface matrix form" is labelled `scatter`.
- Some misses are real: "remove the top and right axis" needs `set_visible` on the spines,
  and the model returned label and tick functions.

So the automatic label is one valid solution plus noise, which caps every method and makes
further tuning against it pointless. Next: a hand-checked test sample where any correct
function counts (`label.py`).

## Model choice (made on validation, before touching test)

Fine-tuned MiniLM, lr 1e-4, 5 epochs (`models/final`): best R@5 on hard questions, which is
the problem this project is about. Scored once on test: see README.
