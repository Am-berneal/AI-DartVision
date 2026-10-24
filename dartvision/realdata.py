"""
Real-data experiments on the DeepDarts throws (data/real_throws.csv, made by prepare_real_data.py).

Experiment 1 - transfer:  models trained on SYNTHETIC points, tested on real held-out sessions.
Experiment 2 - real:      the same 5 models trained on REAL throws, tested on the same held-out sessions.
The split is by session (whole throwing sessions are held out), so darts from one session never
appear in both train and test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import plots
from .evaluate import full_report
from .models import GeometryBaseline, get_models
from .preprocessing import FEATURE_SETS, add_features, feature_matrix, validate_coordinates

CITATION = ("W. McNally et al., \"DeepDarts Dataset,\" IEEE DataPort, 2021, doi: 10.21227/05e7-xs69. "
            "Paper: \"DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts "
            "using a Single Camera,\" CVPR Sports Workshop 2021.")


def load_real(path: Path) -> tuple[pd.DataFrame | None, dict]:
    if not path.exists():
        return None, {}
    raw = pd.read_csv(path)
    real, rep = validate_coordinates(raw)
    real = add_features(real.reset_index(drop=True))
    real["date"] = pd.to_datetime(real["session"].str.slice(3, 13), format="%m_%d_%Y", errors="coerce")
    return real, rep


def split_sessions(real: pd.DataFrame, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(real, groups=real["session"]))
    return real.iloc[tr], real.iloc[te]


def run(real: pd.DataFrame, fitted: dict, seed: int, test_size: float, figs: Path, results: Path,
        log=print) -> list[dict]:
    rows = []
    train, test = split_sessions(real, test_size, seed)
    log(f"Real data: {len(train)} train darts ({train.session.nunique()} sessions), "
        f"{len(test)} test darts ({test.session.nunique()} held-out sessions)")

    geo = GeometryBaseline().fit(None)
    g = full_report(test["x"], test["y"], test["label"], geo.predict(test[["x", "y"]]))
    for ev in ("real", "real_trained"):
        rows.append({"model": "Geometry baseline", "feature_set": "rules", "eval_set": ev,
                     "trained_on": "rules", "train_seconds": 0.0, **g})

    # Experiment 1: synthetic-trained models on real held-out sessions
    for (name, fs), (model, secs) in fitted.items():
        pred = model.predict(feature_matrix(test, fs))
        rows.append({"model": name, "feature_set": fs, "eval_set": "real", "trained_on": "synthetic",
                     "train_seconds": secs, **full_report(test["x"], test["y"], test["label"], pred)})

    # Experiment 2: train on real throws
    for fs in FEATURE_SETS:
        for name, model in get_models(seed).items():
            log(f"Training {name} on REAL throws ('{fs}') ...")
            t0 = time.time()
            model.fit(feature_matrix(train, fs), train["label"])
            secs = round(time.time() - t0, 2)
            pred = model.predict(feature_matrix(test, fs))
            rows.append({"model": name, "feature_set": fs, "eval_set": "real_trained", "trained_on": "real",
                         "train_seconds": secs, **full_report(test["x"], test["y"], test["label"], pred)})

    # Figures
    plots.save(plots.landing_map(real.sample(min(4000, len(real)), random_state=seed),
                                 title="Real DeepDarts throws (sample of 4,000)", s=4), figs / "real_landing_map.png")
    plots.save(plots.density_heatmap(real, bins=60, title="Real dart density (DeepDarts)"), figs / "real_heatmap.png")
    plots.save(plots.region_distribution(real, "Region distribution - real throws"), figs / "real_region_distribution.png")
    sess = session_table(real)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(sess)), sess["avg_score"], marker="o", color="#B83A2B")
    ax.set_xticks(range(len(sess)), [s["session"][3:] for _, s in sess.iterrows()], rotation=90, fontsize=6)
    ax.set_ylabel("average points per dart"); ax.set_title("Real sessions - average score per dart")
    plots.save(fig, figs / "real_session_trend.png")

    m = pd.DataFrame(rows)
    eng = m[(m.feature_set == "engineered")]
    pivot = eng.pivot_table(index="model", columns="eval_set", values="macro_f1")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot[["real", "real_trained"]].rename(columns={"real": "trained on synthetic", "real_trained": "trained on real"}) \
        .drop(index="Geometry baseline", errors="ignore").plot.bar(ax=ax, color=["#9fb7b9", "#B83A2B"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Macro F1 on held-out real sessions"); ax.set_xlabel("")
    ax.set_title("Synthetic vs real training data (engineered features)"); ax.tick_params(axis="x", rotation=15)
    plots.save(fig, figs / "real_transfer_comparison.png")

    summary = real_summary(real, train, test)
    (results / "real_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows


def session_table(real: pd.DataFrame) -> pd.DataFrame:
    g = real.groupby("session")
    t = pd.DataFrame({
        "subset": g["subset"].first(),
        "date": g["date"].first().dt.strftime("%Y-%m-%d"),
        "darts": g.size(),
        "visits": g["visit_id"].nunique(),
        "avg_score": g["score"].mean().round(2),
        "three_dart_avg": (g["score"].mean() * 3).round(1),
        "miss_rate": g["label"].apply(lambda s: (s == "MISS").mean()).round(3),
        "spread_mm": (g["r"].std() * 170).round(1),
    }).reset_index()
    return t.sort_values(["date", "session"]).reset_index(drop=True)


def real_summary(real: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    visits = real.groupby("visit_id")["score"].sum()
    by_dart = real.groupby("dart")["score"].mean().round(2)
    top = real["label"].value_counts().head(10)
    sub = real.groupby("subset").agg(darts=("x", "size"), avg_score=("score", "mean"),
                                     miss_rate=("label", lambda s: (s == "MISS").mean()))
    sample = real.sample(min(3000, len(real)), random_state=0)
    return {
        "citation": CITATION,
        "darts": int(len(real)), "visits": int(real["visit_id"].nunique()), "sessions": int(real["session"].nunique()),
        "date_range": [str(real["date"].min().date()), str(real["date"].max().date())],
        "avg_score_per_dart": round(float(real["score"].mean()), 2),
        "three_dart_avg": round(float(visits.mean()), 1),
        "max_visit": int(visits.max()),
        "visits_100_plus": int((visits >= 100).sum()),
        "miss_rate": round(float((real["label"] == "MISS").mean()), 3),
        "treble_rate": round(float(real["label"].str.startswith("T").mean()), 3),
        "double_rate": round(float(real["label"].str.startswith("D").mean()), 3),
        "bull_rate": round(float(real["label"].isin(["IB", "OB"]).mean()), 3),
        "by_dart_number": {str(k): float(v) for k, v in by_dart.items()},
        "top_regions": {k: int(v) for k, v in top.items()},
        "subsets": {k: {"darts": int(r.darts), "avg_score": round(float(r.avg_score), 2),
                        "miss_rate": round(float(r.miss_rate), 3)} for k, r in sub.iterrows()},
        "split": {"train_darts": int(len(train)), "train_sessions": int(train["session"].nunique()),
                  "test_darts": int(len(test)), "test_sessions": int(test["session"].nunique())},
        "sessions_table": session_table(real).to_dict(orient="records"),
        "sample": sample[["x", "y", "label", "score", "subset", "dart"]].round(4).to_dict(orient="records"),
    }