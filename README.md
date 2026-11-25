# Smart Study Planner: Sentiment Analysis with DistilBERT and LLM Agents

[![tests](https://github.com/ZIADAYMAN81/smart-study-planner-nlp/actions/workflows/tests.yml/badge.svg)](https://github.com/ZIADAYMAN81/smart-study-planner-nlp/actions/workflows/tests.yml)

> Natural Language Processing coursework, The British University in Egypt, 2026. Author: Ziad Ayman.

The natural language processing engine of a study-planning assistant. A student writes how a study session went; a fine-tuned **DistilBERT** transformer detects whether the reflection is positive or negative, and two **large language model agents** (LLaMA 3.3 through Groq) explain the prediction and plan the next session.

```text
$ python -m planner.app "I couldn't focus at all today and the lecture notes made no sense."
Mood: negative (94% confidence)

STRATEGIST:
- Start the next session with a short review of what felt difficult.
- Break the topic into 25-minute blocks with a 5-minute break between them.
- Note one specific question to ask a lecturer or classmate.
```

## Results (balanced IMDB test subset: 1,000 reviews, 500 positive and 500 negative)

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + logistic regression (baseline) | 86.5% | 0.865 |
| DistilBERT, learning rate 5e-5 | 87.5% | 0.875 |
| **DistilBERT, learning rate 2e-5** | **87.8%** | **0.878** |

All models were trained on the same balanced subset of 2,000 reviews (two epochs for DistilBERT, on a laptop CPU). Fine-tuning beats the strong classical baseline, and the lower learning rate is slightly better. Training on the full 25,000 reviews, with a GPU, is the obvious next step and typically lifts DistilBERT above 90%. Full metrics and confusion matrices are in [`results/metrics.json`](results/metrics.json).

## What was fixed and improved in this version
- **Valid evaluation.** The IMDB splits are stored sorted by label, so the original "first 200 / first 100 rows" subsets contained **only negative reviews**, and the reported 100% accuracy was meaningless. Every subset is now shuffled and stratified, and the code checks that both classes are present.
- **A real baseline.** A TF-IDF and logistic regression model on the same data shows how much the transformer actually adds.
- **No secrets in code.** The Groq API key is read from the `GROQ_API_KEY` environment variable (see `.env.example`).
- **Better agent prompts.** Agents now receive the label name and the model's confidence (not a bare `0` or `1`), with role-specific instructions; the strategist acts as a study coach.
- **Works offline.** Without an API key, the planner still classifies the reflection and returns rule-based study suggestions.
- **Command-line app**, reproducible training script, saved models and unit tests.

## How it works
1. **Data:** balanced, shuffled subsets of the IMDB review dataset (Hugging Face `datasets`).
2. **Model:** `distilbert-base-uncased` fine-tuned for two-class classification with the Hugging Face `Trainer` (weight decay, warm-up, dynamic padding), comparing two learning rates.
3. **Agents:** the *analyst* explains which phrases drove the prediction; the *strategist* turns the detected mood into a concrete plan for the next study session.

## Getting started
```bash
pip install -r requirements.txt

python -m planner.train                       # baseline + two DistilBERT runs (about 50 minutes on CPU)
python -m planner.app "Today went really well, I finished all my revision."

export GROQ_API_KEY=...                       # optional: enables the LLM agents
python -m pytest -q
```

## Project structure
```
planner/
├── data.py       # balanced, shuffled IMDB subsets
├── baseline.py   # TF-IDF + logistic regression
├── train.py      # DistilBERT fine-tuning and comparison (command line)
├── agents.py     # analyst and strategist agents (Groq / LLaMA 3.3, offline fallback)
├── app.py        # Smart Study Planner command line
└── metrics.py
results/          # metrics and confusion matrices
tests/
```

## Tech stack
Python · PyTorch · Hugging Face Transformers and Datasets · DistilBERT · Groq (LLaMA 3.3) · scikit-learn · pytest

## License
MIT, see [LICENSE](LICENSE).
