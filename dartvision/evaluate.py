"""Evaluation: classification metrics, score-level metrics, boundary checks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from .board import LABELS, distance_to_boundary, labels_to_scores

BOUNDARY_EPS = 0.01  # ~1.7 mm from a wire counts as a "boundary" throw


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def score_metrics(y_true, y_pred) -> dict:
    st, sp = labels_to_scores(y_true), labels_to_scores(y_pred)
    return {
        "score_accuracy": float(np.mean(st == sp)),
        "score_mae": float(np.mean(np.abs(st - sp))),
    }


def boundary_accuracy(x, y, y_true, y_pred, eps: float = BOUNDARY_EPS) -> dict:
    d = distance_to_boundary(np.asarray(x), np.asarray(y))
    near = d < eps
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "n_boundary": int(near.sum()),
        "boundary_accuracy": float(np.mean(y_true[near] == y_pred[near])) if near.any() else np.nan,
        "interior_accuracy": float(np.mean(y_true[~near] == y_pred[~near])) if (~near).any() else np.nan,
    }


def full_report(x, y, y_true, y_pred) -> dict:
    out = {}
    out.update(classification_metrics(y_true, y_pred))
    out.update(score_metrics(y_true, y_pred))
    out.update(boundary_accuracy(x, y, y_true, y_pred))
    return out


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=LABELS)