"""Evaluation metrics (NumPy only, so they are tested in CI without a GPU stack).

Labels are 0 intact, 1 damaged, 2 destroyed (BRIGHT's 1/2/3 minus one).
"""

from __future__ import annotations

import numpy as np

CLASSES = ("intact", "damaged", "destroyed")


def confusion(y: np.ndarray, pred: np.ndarray, n: int = 3) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n, n), np.int64)
    np.add.at(cm, (np.asarray(y), np.asarray(pred)), 1)
    return cm


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top-1 prediction (equal-width confidence bins)."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature that minimises validation negative log-likelihood (1-D search)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))
    nll = [-np.log(softmax(logits, t)[np.arange(len(y)), y] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def damage_score(probs: np.ndarray) -> np.ndarray:
    """Expected severity in [0, 2]; used to rank buildings for a first look."""
    return probs @ np.array([0.0, 1.0, 2.0])


def recall_at_fraction(
    score: np.ndarray, y: np.ndarray, target: int = 2, frac: float = 0.1
) -> float:
    """Share of ``target``-class buildings found by inspecting the top ``frac`` of buildings."""
    k = max(1, int(round(frac * len(score))))
    top = np.argsort(-score)[:k]
    pos = (y == target).sum()
    return float((y[top] == target).sum() / pos) if pos else float("nan")


def binary_auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve via the rank statistic (ties get average ranks)."""
    positive = np.asarray(positive, bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = np.asarray(score)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_bias(logits: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
    """Per-class logit offsets (intact fixed at 0) that maximise macro-F1 on validation data.

    Arg-max of a model trained on 95 % intact buildings under-predicts the rare classes; shifting
    the decision boundary on the validation city is the standard, test-blind correction.
    """
    grid = np.linspace(-1.0, 4.0, 26) if grid is None else grid
    best, out = -1.0, np.zeros(3)
    for b1 in grid:
        for b2 in grid:
            bias = np.array([0.0, b1, b2])
            f = per_class_f1(confusion(y, (logits + bias).argmax(1))).mean()
            if f > best:
                best, out = f, bias
    return out


def report(
    logits: np.ndarray, y: np.ndarray, temperature: float = 1.0, bias: np.ndarray | None = None
) -> dict:
    probs = softmax(logits, temperature)
    pred = (logits + (0 if bias is None else bias)).argmax(1)
    cm = confusion(y, pred)
    f1 = per_class_f1(cm)
    score = damage_score(probs)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1.mean()),
        "f1": {c: float(v) for c, v in zip(CLASSES, f1)},
        "confusion": cm.tolist(),
        "ece": ece(probs, y),
        "temperature": temperature,
        "bias": None if bias is None else [float(b) for b in bias],
        "auroc_damaged_or_destroyed": binary_auroc(score, y > 0),
        "destroyed_recall_top10pct": recall_at_fraction(score, y, 2, 0.10),
    }
