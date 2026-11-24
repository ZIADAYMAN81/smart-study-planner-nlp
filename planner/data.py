"""IMDB loading with balanced, shuffled subsets.

The IMDB splits are stored sorted by label (all negative reviews first), so taking the first N
rows yields a single class. Every subset here is shuffled and stratified, and checked to contain
both classes.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

LABEL_NAMES = ["negative", "positive"]


def balanced_indices(labels: List[int], n: int, seed: int = 42) -> List[int]:
    """Pick ``n`` indices with (as close as possible to) equal numbers of each class."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = n // len(classes)
    chosen = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        if len(idx) < per_class:
            raise ValueError(f"class {c} has only {len(idx)} examples, {per_class} requested")
        chosen.extend(rng.choice(idx, size=per_class, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def load_imdb_subsets(n_train: int = 2000, n_test: int = 1000, seed: int = 42) -> Tuple[dict, dict]:
    """Return ``{"text": [...], "label": [...]}`` dictionaries for balanced train and test subsets."""
    from datasets import load_dataset

    ds = load_dataset("imdb")
    out = []
    for split, n in (("train", n_train), ("test", n_test)):
        labels = ds[split]["label"]
        idx = balanced_indices(labels, n, seed)
        subset = ds[split].select(idx)
        assert len(set(subset["label"])) == 2, "subset must contain both classes"
        out.append({"text": list(subset["text"]), "label": list(subset["label"])})
    return out[0], out[1]
