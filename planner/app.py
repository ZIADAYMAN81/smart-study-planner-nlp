"""Smart Study Planner command line: ``python -m planner.app "I couldn't focus at all today."``.

Classifies the mood of a study reflection with the fine-tuned DistilBERT model, then asks the
analyst and strategist agents to explain the result and plan the next session.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .agents import Agent, Prediction


class SentimentModel:
    def __init__(self, model_dir: str):
        from transformers import pipeline

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} not found; run `python -m planner.train` first")
        self.pipe = pipeline("text-classification", model=model_dir, tokenizer=model_dir, truncation=True)

    def __call__(self, text: str) -> Prediction:
        out = self.pipe(text)[0]
        return Prediction(label=out["label"].lower(), confidence=float(out["score"]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Analyse a study reflection and plan the next session.")
    ap.add_argument("text", help="how the study session went, in your own words")
    ap.add_argument("--model", default="models/distilbert_lr2e-05", help="folder saved by planner.train")
    ap.add_argument("--no-agents", action="store_true", help="only print the sentiment")
    args = ap.parse_args(argv)

    pred = SentimentModel(args.model)(args.text)
    print(f"Mood: {pred.label} ({pred.confidence:.0%} confidence)")
    if args.no_agents:
        return
    for role in ("analyst", "strategist"):
        agent = Agent(role)
        print(f"\n{role.upper()}{'' if agent.online else ' (offline)'}:\n{agent.respond(args.text, pred)}")


if __name__ == "__main__":
    main()
