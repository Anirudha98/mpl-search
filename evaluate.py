"""Score retrieval methods on the held-out Stack Overflow questions.

A question counts as found at k if any of its labelled functions is in the top k results.
Reported on all test questions and on the "hard" ones, where no labelled function's name
appears in the title, so word overlap cannot help.

Differences between methods come with a 95% paired bootstrap interval over questions.

Usage:  .venv/bin/python evaluate.py [--split test] [--model models/finetuned]
        .venv/bin/python evaluate.py --check
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
TOKEN_RE = re.compile(r"[a-z0-9]+")
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load(split):
    corpus = [json.loads(l) for l in (ROOT / "data" / "corpus.jsonl").open()]
    qs = [json.loads(l) for l in (ROOT / "data" / "questions.jsonl").open()]
    return corpus, [q for q in qs if q["split"] == split]


def tokens(text):
    # set_xlim -> set, xlim: lets "xlim" in a question meet set_xlim in the docs
    return TOKEN_RE.findall(text.lower().replace("_", " "))


def is_hard(q):
    title = q["title"].lower()
    return not any(n.lower() in title or n.lower().replace("_", " ") in title for n in q["labels"])


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [Counter(tokens(d)) for d in docs]
        self.lens = np.array([sum(t.values()) for t in self.toks])
        self.avg = self.lens.mean()
        df = Counter(w for t in self.toks for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        s = np.zeros(len(self.toks))
        for w in set(tokens(query)):
            if w not in self.idf:
                continue
            tf = np.array([t.get(w, 0) for t in self.toks])
            s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens / self.avg))
        return s


def rank_bm25(corpus, queries):
    bm = BM25([d["text"] for d in corpus])
    return [np.argsort(-bm.scores(q))[:10] for q in queries]


def rank_dense(corpus, queries, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    d = model.encode([c["text"] for c in corpus], normalize_embeddings=True, batch_size=64)
    q = model.encode(queries, normalize_embeddings=True, batch_size=64)
    return [np.argsort(-row)[:10] for row in q @ d.T]


def hits(ranked, corpus, qs, k):
    names = [c["name"] for c in corpus]
    return np.array([any(names[i] in q["labels"] for i in r[:k]) for r, q in zip(ranked, qs)], dtype=float)


def mrr(ranked, corpus, qs):
    names = [c["name"] for c in corpus]
    out = []
    for r, q in zip(ranked, qs):
        ranks = [i for i, d in enumerate(r[:10]) if names[d] in q["labels"]]
        out.append(1 / (ranks[0] + 1) if ranks else 0.0)
    return np.array(out)


def paired_ci(a, b, n=2000, seed=0):
    """95% bootstrap interval for mean(b - a), resampling questions."""
    rng = np.random.default_rng(seed)
    diff = b - a
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return diff.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def report(results, qs):
    hard = np.array([is_hard(q) for q in qs])
    print(f"{len(qs)} questions, {hard.sum()} hard (no label name in the title)\n")
    print(f"{'method':<28}{'R@1':>7}{'R@5':>7}{'MRR@10':>8}{'R@5 hard':>10}")
    for name, (r1, r5, m) in results.items():
        print(f"{name:<28}{r1.mean():>7.2f}{r5.mean():>7.2f}{m.mean():>8.2f}{r5[hard].mean():>10.2f}")
    names = list(results)
    for ref in names[:2]:  # vs keyword search, and vs the embedding model before fine-tuning
        print(f"\nR@5 difference vs {ref}, 95% paired bootstrap interval:")
        for name in names[names.index(ref) + 1:]:
            for label, mask in (("all", slice(None)), ("hard", hard)):
                d, lo, hi = paired_ci(results[ref][1][mask], results[name][1][mask])
                print(f"  {name:<26}{label:<6}{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]")


def check():
    corpus = [{"name": "legend", "text": "legend place a legend on the axes"},
              {"name": "set_xlim", "text": "set_xlim set the x axis view limits"}]
    qs = [{"title": "how do I change the x limits", "labels": ["set_xlim"]},
          {"title": "move the legend outside", "labels": ["legend"]}]
    ranked = rank_bm25(corpus, [q["title"] for q in qs])
    assert hits(ranked, corpus, qs, 1).tolist() == [1.0, 1.0], ranked
    assert mrr([np.array([0, 1])], corpus, qs[:1]).tolist() == [0.5]
    assert is_hard({"title": "change limits", "labels": ["set_xlim"]})
    assert not is_hard({"title": "set xlim fails", "labels": ["set_xlim"]})
    d, lo, hi = paired_ci(np.zeros(100), np.ones(100))
    assert d == lo == hi == 1.0
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", default=[], help="extra embedding model(s) to score")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        sys.exit()
    corpus, qs = load(args.split)
    queries = [q["title"] for q in qs]
    methods = {"BM25": lambda: rank_bm25(corpus, queries),
               "MiniLM (off the shelf)": lambda: rank_dense(corpus, queries, BASE_MODEL)}
    for m in args.model:
        methods[Path(m).name] = lambda m=m: rank_dense(corpus, queries, m)
    results = {}
    for name, fn in methods.items():
        ranked = fn()
        results[name] = (hits(ranked, corpus, qs, 1), hits(ranked, corpus, qs, 5), mrr(ranked, corpus, qs))
    report(results, qs)
