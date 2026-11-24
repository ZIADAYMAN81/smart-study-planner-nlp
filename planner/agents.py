"""LLM reasoning agents (Groq, LLaMA 3.3) that explain predictions and suggest next steps.

The API key is read from the ``GROQ_API_KEY`` environment variable; it is never stored in code.
Without a key, deterministic offline responses are returned so the planner still works.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

MODEL = "llama-3.3-70b-versatile"

ROLES = {
    "analyst": ("You are an NLP analyst. In 3 short bullet points, explain which words or phrases most likely "
                "led a sentiment classifier to its prediction, and whether the prediction seems right.", 0.3),
    "strategist": ("You are a supportive study coach. Given a student's reflection on a study session and its "
                   "detected sentiment, suggest a concrete plan for the next session in 3 short bullet points.", 0.5),
}


@dataclass
class Prediction:
    label: str          # "positive" or "negative"
    confidence: float   # probability of the predicted label, 0-1


def build_messages(role: str, text: str, pred: Prediction) -> List[dict]:
    if role not in ROLES:
        raise ValueError(f"unknown agent role {role!r}; choose from {sorted(ROLES)}")
    system, _ = ROLES[role]
    user = (f"Text: {text}\n"
            f"Predicted sentiment: {pred.label} (confidence {pred.confidence:.0%})")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def offline_response(role: str, pred: Prediction) -> str:
    if role == "analyst":
        return (f"Offline mode (set GROQ_API_KEY for an LLM explanation): the classifier labelled this text "
                f"{pred.label} with {pred.confidence:.0%} confidence.")
    if pred.label == "negative":
        return ("- Start the next session with a short review of what felt difficult.\n"
                "- Break the topic into 25-minute blocks with a 5-minute break between them.\n"
                "- Note one specific question to ask a lecturer or classmate.")
    return ("- Keep the same routine; it is working.\n"
            "- Add a slightly harder topic or practice questions next time.\n"
            "- Spend 10 minutes summarising today's material to lock it in.")


class Agent:
    def __init__(self, role: str, api_key: Optional[str] = None):
        if role not in ROLES:
            raise ValueError(f"unknown agent role {role!r}")
        self.role = role
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")

    @property
    def online(self) -> bool:
        return bool(self.api_key)

    def respond(self, text: str, pred: Prediction) -> str:
        if not self.online:
            return offline_response(self.role, pred)
        from groq import Groq

        client = Groq(api_key=self.api_key)
        res = client.chat.completions.create(model=MODEL, messages=build_messages(self.role, text, pred),
                                             temperature=ROLES[self.role][1])
        return res.choices[0].message.content
