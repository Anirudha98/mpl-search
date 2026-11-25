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
