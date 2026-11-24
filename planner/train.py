"""Fine-tune and compare DistilBERT configurations: ``python -m planner.train --lrs 5e-5 2e-5``.

Trains each learning rate on the same balanced training subset, evaluates on a balanced,
held-out test subset, compares against the TF-IDF baseline and writes ``results/metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .baseline import train_baseline
from .data import load_imdb_subsets
from .metrics import evaluate

MODEL_NAME = "distilbert-base-uncased"


def fine_tune(train: dict, test: dict, lr: float, epochs: int, batch_size: int, max_length: int,
              seed: int, save_dir: Path | None = None) -> dict:
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(d):
        enc = tok(d["text"], truncation=True, max_length=max_length)
        return Dataset.from_dict({**enc, "labels": d["label"]})

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1})
    args = TrainingArguments(output_dir=str(save_dir or "outputs/tmp"), learning_rate=lr,
                             per_device_train_batch_size=batch_size, per_device_eval_batch_size=32,
                             num_train_epochs=epochs, weight_decay=0.01, warmup_ratio=0.1,
                             eval_strategy="no", save_strategy="no", logging_steps=25, report_to=[], seed=seed)
    from transformers import DataCollatorWithPadding

    trainer = Trainer(model=model, args=args, train_dataset=encode(train),
                      data_collator=DataCollatorWithPadding(tok))
    start = time.time()
    trainer.train()
    logits = trainer.predict(encode(test)).predictions
    metrics = evaluate(test["label"], np.argmax(logits, axis=1))
    metrics.update({"model": f"DistilBERT (learning rate {lr:g})", "learning_rate": lr, "epochs": epochs,
                    "train_size": len(train["label"]), "train_minutes": round((time.time() - start) / 60, 1)})
    if save_dir:
        trainer.save_model(str(save_dir))
        tok.save_pretrained(str(save_dir))
    del trainer, model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return metrics


def main(argv=None) -> list:
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT for review sentiment and compare learning rates.")
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-5, 2e-5])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default="models", help="best model is saved here as models/best")
    ap.add_argument("--results", default="results/metrics.json")
    args = ap.parse_args(argv)

    train, test = load_imdb_subsets(args.n_train, args.n_test, args.seed)
    print(f"train: {len(train['label'])} reviews ({sum(train['label'])} positive) | "
          f"test: {len(test['label'])} reviews ({sum(test['label'])} positive)", flush=True)

    results = [train_baseline(train, test)]
    print(json.dumps(results[-1]), flush=True)
    best = None
    for lr in args.lrs:
        save = Path(args.models_dir) / f"distilbert_lr{lr:g}"
        m = fine_tune(train, test, lr, args.epochs, args.batch_size, args.max_length, args.seed, save)
        m["saved_to"] = str(save)
        results.append(m)
        print(json.dumps(m), flush=True)
        if best is None or m["accuracy"] > best["accuracy"]:
            best = m

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps({"results": results, "best": best["model"]}, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
