"""
Train and evaluate the same 5 classifiers (raw vs engineered features) on archery and shooting targets.

Run:  python train_sports.py          (run_pipeline.py also calls this at the end)
Outputs:
    models/sports/<target>__<model>_<features>.joblib
    results/sports_metrics.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from dartvision import sports
from dartvision.models import get_models
from dartvision.preprocessing import FEATURE_SETS, add_features

ROOT = Path(__file__).parent
OUT_MODELS = ROOT / "models" / "sports"
RESULTS = ROOT / "results"


def model_key(name: str, fs: str) -> str:
    return f"{name.lower().replace(' ', '_')}_{fs}"


def features(p: sports.TargetProfile, df: pd.DataFrame, fs: str) -> pd.DataFrame:
    nx, ny = sports.normalised(p, df["x"], df["y"])
    return add_features(pd.DataFrame({"x": nx, "y": ny}))[FEATURE_SETS[fs]]


def train_all(n_per_class: int = 300, seed: int = 42, log=print) -> pd.DataFrame:
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sports.PROFILES.values():
        df = sports.make_dataset(p, n_per_class=n_per_class, seed=seed)
        train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
        bd = sports.boundary_distances(p, np.hypot(test["x"], test["y"]))
        near = bd < 0.1 * p.ring_width_mm                       # within 10% of a ring width of a scoring line
        log(f"[sports] {p.name}: {len(train)} train / {len(test)} test, {df['label'].nunique()} classes")
        for fs in FEATURE_SETS:
            for name, model in get_models(seed).items():
                t0 = time.time()
                model.fit(features(p, train, fs), train["label"])
                secs = time.time() - t0
                pred = model.predict(features(p, test, fs))
                yt = test["label"].values
                rows.append({
                    "target": p.key, "target_name": p.name, "model": name, "feature_set": fs,
                    "accuracy": accuracy_score(yt, pred),
                    "macro_f1": f1_score(yt, pred, average="macro", zero_division=0),
                    "boundary_accuracy": float(np.mean(yt[near] == pred[near])) if near.any() else np.nan,
                    "n_boundary": int(near.sum()), "n_test": len(test), "train_seconds": round(secs, 2),
                })
                joblib.dump({"model": model, "feature_set": fs, "name": name, "target": p.key},
                            OUT_MODELS / f"{p.key}__{model_key(name, fs)}.joblib", compress=3)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sports_metrics.csv", index=False)
    best = out.sort_values(["target", "macro_f1"], ascending=[True, False]).groupby("target").head(1)
    log("[sports] Best model per target:\n" + best[["target", "model", "feature_set", "accuracy", "boundary_accuracy"]]
        .round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    train_all()