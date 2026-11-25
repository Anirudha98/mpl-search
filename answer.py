"""Answer a question from the top 3 retrieved functions, with citations, as structured JSON.

The model must reply with exactly:
    {"found": true|false, "answer": "...", "citations": [1, 3]}
citations index the excerpts it was shown. If the excerpts do not answer the question it must
say found=false instead of guessing. Replies that are not valid JSON, or that cite an excerpt
that was not shown, are counted, not hidden.

Needs NVIDIA_API_KEY in the environment or in .env (free key from build.nvidia.com).

Usage:  .venv/bin/python answer.py "how do I move the legend outside the plot"
        .venv/bin/python answer.py --eval 100        # score on the first 100 test questions
        .venv/bin/python answer.py --check           # parsing and scoring, no API calls
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).parent
API = "https://integrate.api.nvidia.com/v1/chat/completions"
LLM = "meta/llama-3.3-70b-instruct"
MODEL_DIR = ROOT / "models" / "final"
K = 3

SYSTEM = """You answer matplotlib questions using ONLY the numbered documentation excerpts given.
Reply with one JSON object and nothing else:
{"found": true or false, "answer": "<2-4 sentences, name the function to call>", "citations": [<excerpt numbers you used>]}
If the excerpts do not contain what is needed, reply {"found": false, "answer": "", "citations": []}.
Never cite an excerpt number that was not given."""


def key():
    if os.environ.get("NVIDIA_API_KEY"):
        return os.environ["NVIDIA_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "NVIDIA_API_KEY":
                return value.strip().strip("\"'")
    sys.exit("No NVIDIA_API_KEY. Put a line NVIDIA_API_KEY=... in .env")


class Retriever:
    def __init__(self, model_dir=MODEL_DIR):
        from sentence_transformers import SentenceTransformer
        self.corpus = [json.loads(l) for l in (ROOT / "data" / "corpus.jsonl").open()]
        self.model = SentenceTransformer(str(model_dir))
        self.emb = self.model.encode([d["text"] for d in self.corpus], normalize_embeddings=True)

    def top(self, question, k=K):
        q = self.model.encode([question], normalize_embeddings=True)[0]
        return [self.corpus[i] for i in (-(self.emb @ q)).argsort()[:k]]


def prompt(question, docs):
    excerpts = "\n\n".join(f"[{i}] {d['text'][:800]}" for i, d in enumerate(docs, 1))
    return f"Excerpts:\n{excerpts}\n\nQuestion: {question}"


def ask(question, docs, api_key):
    r = requests.post(API, timeout=60, headers={"Authorization": f"Bearer {api_key}"}, json={
        "model": LLM, "temperature": 0, "max_tokens": 400,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(question, docs)}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse(reply, n_docs):
    """-> (dict, problem). problem is None, 'not_json' or 'bad_citation'."""
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", reply.strip())  # models often fence JSON anyway
    try:
        out = json.loads(cleaned)
        assert isinstance(out, dict) and isinstance(out.get("citations"), list)
    except (json.JSONDecodeError, AssertionError):
        return None, "not_json"
    if any(not isinstance(c, int) or not 1 <= c <= n_docs for c in out["citations"]):
        return out, "bad_citation"
    return out, None


def score(rows):
    """rows: (question labels, retrieved docs, parsed reply, problem)."""
    n = len(rows)
    s = {"questions": n, "retrieved_correct": 0, "not_json": 0, "bad_citation": 0,
         "said_not_found": 0, "cites_correct": 0}
    for labels, docs, out, problem in rows:
        if any(d["name"] in labels for d in docs):
            s["retrieved_correct"] += 1
        if problem:
            s[problem] += 1
            continue
        if not out.get("found"):
            s["said_not_found"] += 1
        elif any(docs[c - 1]["name"] in labels for c in out["citations"]):
            s["cites_correct"] += 1
    return s


def evaluate(n):
    retriever, api_key = Retriever(), key()
    qs = [json.loads(l) for l in (ROOT / "data" / "questions.jsonl").open()]
    qs = [q for q in qs if q["split"] == "test"][:n]
    rows, log = [], (ROOT / "data" / "answers_test.jsonl").open("w")
    for q in qs:
        docs = retriever.top(q["title"])
        reply = ask(q["title"], docs, api_key)
        out, problem = parse(reply, len(docs))
        rows.append((q["labels"], docs, out, problem))
        log.write(json.dumps({"id": q["id"], "title": q["title"], "labels": q["labels"],
                              "retrieved": [d["name"] for d in docs], "reply": reply, "problem": problem}) + "\n")
    s = score(rows)
    print(json.dumps(s, indent=2))
    print(f"cites a correct function: {s['cites_correct'] / n:.0%} of questions, "
          f"{s['cites_correct'] / max(s['retrieved_correct'], 1):.0%} of those where retrieval found one")


def check():
    assert parse('```json\n{"found": true, "answer": "x", "citations": [2]}\n```', 3)[1] is None
    assert parse('{"found": true, "answer": "x", "citations": [4]}', 3)[1] == "bad_citation"
    assert parse("Sure! Use legend()", 3)[1] == "not_json"
    docs = [{"name": "legend"}, {"name": "grid"}, {"name": "text"}]
    s = score([(["legend"], docs, {"found": True, "citations": [1]}, None),
               (["legend"], docs, {"found": True, "citations": [2]}, None),
               (["axvspan"], docs, {"found": False, "citations": []}, None),
               (["legend"], docs, None, "not_json")])
    assert (s["retrieved_correct"], s["cites_correct"], s["said_not_found"], s["not_json"]) == (3, 1, 1, 1), s
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--eval", type=int)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
    elif args.eval:
        evaluate(args.eval)
    elif args.question:
        docs = Retriever().top(args.question)
        out, problem = parse(ask(args.question, docs, key()), len(docs))
        print(json.dumps({"retrieved": [d["name"] for d in docs], "reply": out, "problem": problem}, indent=2))
    else:
        ap.print_help()
