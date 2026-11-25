"""Fine-tune the embedding model on (question title, documentation of a function its answer added).

Training uses only the train split. Loss is in-batch negatives (MultipleNegativesRankingLoss):
each title must score its own document above every other document in the batch. Batches never
hold the same document twice, or a correct pair would be scored as a negative.

Usage:  .venv/bin/python finetune.py [--epochs 3] [--out models/finetuned]
        .venv/bin/python finetune.py --check     (pair building only, no training)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def pairs(corpus, questions):
    text = {d["name"]: d["text"] for d in corpus}
    return [(q["title"], text[n]) for q in questions if q["split"] == "train" for n in q["labels"]]


def train(epochs, out, batch_size, seed, lr):
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    corpus = [json.loads(l) for l in (ROOT / "data" / "corpus.jsonl").open()]
    qs = [json.loads(l) for l in (ROOT / "data" / "questions.jsonl").open()]
    p = pairs(corpus, qs)
    print(f"{len(p)} training pairs")
    model = SentenceTransformer(BASE_MODEL)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(ROOT / "models" / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr, warmup_steps=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no", logging_steps=50,
        seed=seed, report_to="none")
    trainer = SentenceTransformerTrainer(
        model=model, args=args,
        train_dataset=Dataset.from_dict({"anchor": [a for a, _ in p], "positive": [b for _, b in p]}),
        loss=losses.MultipleNegativesRankingLoss(model))
    trainer.train()
    model.save(out)
    print(f"saved to {out}")


def check():
    corpus = [{"name": "legend", "text": "legend doc"}, {"name": "grid", "text": "grid doc"}]
    qs = [{"title": "t1", "labels": ["legend", "grid"], "split": "train"},
          {"title": "t2", "labels": ["grid"], "split": "test"}]
    assert pairs(corpus, qs) == [("t1", "legend doc"), ("t1", "grid doc")]  # test split never trains
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default=str(ROOT / "models" / "finetuned"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        sys.exit()
    train(args.epochs, args.out, args.batch_size, args.seed, args.lr)
