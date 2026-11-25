"""Collect Stack Overflow matplotlib questions and label each with the functions its answer introduced.

The query is the question title: that is the everyday wording people search with.
The label is every matplotlib function called in the accepted answer's code that the question's
own code did not already call. "What did the answer add" is the function the asker was missing.

Two kinds of name are dropped from labels:
- scaffolding: many answers rebuild the whole plot, so figure/subplots/plot/show show up as
  "added" without being what the asker was missing;
- names that collide with Python itself (dict.get, dict.update, ...), so a match means nothing.
Only method-style calls count (ax.x(, plt.x(, fig.x(): a bare set( is Python's set.

Raw API responses are cached in data/cache/, so re-running never spends quota twice.
Stack Overflow content is CC BY-SA 4.0; every row keeps its question link.

Usage:  .venv/bin/python collect.py [--pages 25]   -> data/questions.jsonl
        .venv/bin/python collect.py --check
"""
import argparse
import hashlib
import html
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "data" / "questions.jsonl"
API = "https://api.stackexchange.com/2.3"
SCAFFOLD = {"show", "plot", "subplots", "figure", "add_subplot", "subplot", "gca", "gcf", "close"}
AMBIGUOUS = {"get", "update", "copy", "clear", "format", "index", "pop", "remove", "items"}

CODE_RE = re.compile(r"<code>(.*?)</code>", re.S)
CALL_RE = re.compile(r"\.(\w+)\s*\(")


def get(path, **params):
    key = hashlib.sha1(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    r = requests.get(f"{API}/{path}", params={"site": "stackoverflow", **params}, timeout=30)
    r.raise_for_status()
    data = r.json()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data))
    time.sleep(data.get("backoff", 0) + 0.2)  # the API asks clients to honour backoff
    return data


def fetch(pages):
    # Without an API key the API stops at page 25, so walk several listings and de-duplicate.
    # The third asks only for questions with an accepted answer, newest first, so every row counts.
    listings = [("questions", {"sort": "votes"}), ("questions", {"sort": "activity"}),
                ("search/advanced", {"sort": "creation", "accepted": "True"})]
    seen, questions, answers = set(), [], {}
    for path, extra in listings:
        for page in range(1, min(pages, 25) + 1):
            data = get(path, tagged="matplotlib", order="desc", pagesize=100, page=page,
                       filter="withbody", **extra)
            for q in data["items"]:
                if q.get("accepted_answer_id") and q["question_id"] not in seen:
                    seen.add(q["question_id"])
                    questions.append(q)
            if not data.get("has_more"):
                break
    ids = [q["accepted_answer_id"] for q in questions]
    for i in range(0, len(ids), 100):
        batch = ";".join(map(str, ids[i:i + 100]))
        for a in get(f"answers/{batch}", pagesize=100, filter="withbody")["items"]:
            answers[a["answer_id"]] = a["body"]
    return questions, answers


def calls(body):
    code = " ".join(html.unescape(c) for c in CODE_RE.findall(body))
    return set(CALL_RE.findall(code))


def label(questions, answers, names):
    drop = SCAFFOLD | AMBIGUOUS
    out = []
    for q in questions:
        body = answers.get(q["accepted_answer_id"])
        if body is None:
            continue
        added = (calls(body) - calls(q["body"])) & names - drop
        if added:
            out.append({"id": q["question_id"], "title": html.unescape(q["title"]), "link": q["link"],
                        "labels": sorted(added), "split": split_of(q["question_id"])})
    return out


def split_of(qid):
    # deterministic by id, so a question never moves between splits when more pages are added
    h = int(hashlib.sha1(str(qid).encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 80 else "test"


def check():
    q = [{"question_id": 1, "accepted_answer_id": 10, "link": "x", "title": "Move legend &amp; outside",
          "body": "<pre><code>plt.plot(x)\nplt.show()</code></pre>"}]
    a = {10: "<code>fig, ax = plt.subplots()\nax.plot(x)\nax.legend(bbox_to_anchor=(1, 1))\n"
             "d.get(1)\nset(x)\nnp.foo(1)</code>"}
    rows = label(q, a, {"plot", "legend", "subplots", "get", "set"})
    # subplots is scaffolding, d.get is a dict, bare set( is Python's, np.foo is not matplotlib
    assert rows[0]["labels"] == ["legend"], rows
    assert rows[0]["title"] == "Move legend & outside"
    assert split_of(1) == split_of(1) and {split_of(i) for i in range(300)} == {"train", "val", "test"}
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=25, help="per ordering, at most 25")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        sys.exit()
    names = {json.loads(l)["name"] for l in (ROOT / "data" / "corpus.jsonl").open()}
    questions, answers = fetch(args.pages)
    rows = label(questions, answers, names)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(questions)} questions with an accepted answer, {len(rows)} with a label")
    print("split sizes:", dict(Counter(r["split"] for r in rows)))
