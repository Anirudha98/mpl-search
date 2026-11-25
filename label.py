"""Hand-check a sample of test questions: which candidate functions would actually solve it?

The automatic labels record one solution per question plus incidental calls (EXPERIMENTS.md, 6).
Here a person judges a pool of candidates per question: the top 5 of every method plus the
automatic labels, shuffled, with no sign of which method proposed which. Any candidate marked
correct counts as a hit; unjudged functions count as misses (standard pooled evaluation, so a
method is never rewarded for a function nobody looked at).

Step 1, once:   .venv/bin/python label.py --pool 100     -> data/pool_test.jsonl (runs the models)
Step 2:         .venv/bin/python label.py                -> data/gold.jsonl (resumes where it stopped)
Step 3:         .venv/bin/python evaluate.py --gold
                .venv/bin/python label.py --check
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
POOL = ROOT / "data" / "pool_test.jsonl"
GOLD = ROOT / "data" / "gold.jsonl"
METHODS = {"bm25": None, "minilm": "sentence-transformers/all-MiniLM-L6-v2",
           "final": str(ROOT / "models" / "final"), "bge-base": "BAAI/bge-base-en-v1.5"}


def build_pool(n, seed=0):
    import evaluate as E
    corpus, qs = E.load("test")
    qs = random.Random(seed).sample(qs, n)
    names = [c["name"] for c in corpus]
    titles = [q["title"] for q in qs]
    tops = {}
    for m, model in METHODS.items():
        ranked = E.rank_bm25(corpus, titles) if model is None else E.rank_dense(corpus, titles, model)
        tops[m] = [[names[i] for i in r[:5]] for r in ranked]
    with POOL.open("w") as f:
        for j, q in enumerate(qs):
            cands = sorted({n for m in METHODS for n in tops[m][j]} | set(q["labels"]))
            random.Random(q["id"]).shuffle(cands)
            f.write(json.dumps({"id": q["id"], "title": q["title"], "link": q["link"],
                                "candidates": cands}) + "\n")
    print(f"wrote {n} questions to {POOL}")


def parse_choice(text, n):
    """'1 4 7' -> [0, 3, 6]; 'n' -> []; anything else -> None (ask again)."""
    text = text.strip().lower()
    if text == "n":
        return []
    try:
        picks = [int(t) - 1 for t in text.replace(",", " ").split()]
    except ValueError:
        return None
    return picks if picks and all(0 <= p < n for p in picks) else None


def label():
    corpus = {d["name"]: d["text"] for d in map(json.loads, (ROOT / "data" / "corpus.jsonl").open())}
    done = {json.loads(l)["id"] for l in GOLD.open()} if GOLD.exists() else set()
    pool = [json.loads(l) for l in POOL.open()]
    todo = [p for p in pool if p["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go. For each question type the numbers of every function\n"
          "that would solve it (e.g. '2 5'), 'n' for none, 's' to skip, 'q' to quit.\n")
    with GOLD.open("a") as out:
        for k, p in enumerate(todo, 1):
            print(f"[{len(done) + k}/{len(pool)}] {p['title']}\n    {p['link']}")
            for i, c in enumerate(p["candidates"], 1):
                first = corpus[c].split("\n")[1] if "\n" in corpus[c] else ""
                print(f"  {i:>2}. {c:<28} {first[:80]}")
            while True:
                a = input("> ").strip().lower()
                if a in ("q", "s"):
                    break
                picks = parse_choice(a, len(p["candidates"]))
                if picks is not None:
                    break
                print("  numbers from the list, 'n', 's' or 'q'")
            if a == "q":
                return
            if a == "s":
                continue
            out.write(json.dumps({"id": p["id"], "correct": [p["candidates"][i] for i in picks],
                                  "judged": p["candidates"]}) + "\n")
            out.flush()
            print()


def check():
    assert parse_choice("1 3", 4) == [0, 2]
    assert parse_choice("1,3", 4) == [0, 2]
    assert parse_choice("n", 4) == []
    assert parse_choice("5", 4) is None and parse_choice("abc", 4) is None and parse_choice("", 4) is None
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, help="build a pool of this many test questions")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
    elif args.pool:
        build_pool(args.pool)
    else:
        label()
