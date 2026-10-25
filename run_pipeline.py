"""
AI DartVision - full pipeline.

  1. Data collection   : synthetic geometry-labelled coordinates + simulated players
  2. Preprocessing     : validation, normalisation
  3. EDA + features    : figures, radius/angle features
  4. Model training    : 5 classifiers x 2 feature sets, same split
  5. Evaluation        : held-out metrics, confusion matrices, boundary checks
  6. Outputs           : models/, results/, data/  (used by the Streamlit app)

Run:  python run_pipeline.py            (default settings)
      python run_pipeline.py --n-per-class 400 --seed 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from dartvision import plots, realdata
from dartvision.data import make_synthetic_dataset, make_uniform_dataset, simulate_players
from dartvision.evaluate import confusion, full_report
from dartvision.models import GeometryBaseline, get_models
from dartvision.players import bias_description, player_summary, session_trend
from dartvision.preprocessing import FEATURE_SETS, add_features, feature_matrix, validate_coordinates

ROOT = Path(__file__).parent
DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"
FIGS = RESULTS / "figures"


def log(msg):
    print(f"[DartVision] {msg}", flush=True)


def main(n_per_class: int, seed: int, test_size: float):
    for d in (DATA, MODELS, RESULTS, FIGS):
        d.mkdir(parents=True, exist_ok=True)

    # 1. DATA COLLECTION ------------------------------------------------------
    log("Generating synthetic geometry-labelled dataset ...")
    df = make_synthetic_dataset(n_per_class=n_per_class, seed=seed)
    uniform = make_uniform_dataset(n=5000, seed=seed + 1)
    throws = simulate_players(seed=seed + 2)

    # 2. PREPROCESSING --------------------------------------------------------
    df, report = validate_coordinates(df)
    log(f"Validation report: {report}")
    df = add_features(df)
    uniform = add_features(validate_coordinates(uniform)[0])
    df.to_csv(DATA / "synthetic_coordinates.csv", index=False)
    uniform.to_csv(DATA / "uniform_test_coordinates.csv", index=False)
    throws.to_csv(DATA / "player_throws.csv", index=False)
    log(f"Saved {len(df)} stratified points, {len(uniform)} uniform points, {len(throws)} player throws")

    # 3. EDA ------------------------------------------------------------------
    log("Creating EDA figures ...")
    plots.save(plots.landing_map(df.sample(2000, random_state=seed), title="Synthetic landing coordinates (sample)", s=5), FIGS / "eda_landing_coordinates.png")
    plots.save(plots.region_distribution(df, "Region distribution - stratified training data"), FIGS / "eda_region_distribution.png")
    plots.save(plots.region_distribution(uniform, "Region distribution - uniform board coverage"), FIGS / "eda_region_distribution_uniform.png")
    plots.save(plots.score_distribution(uniform, "Score distribution - uniform board coverage"), FIGS / "eda_score_distribution.png")
    plots.save(plots.density_heatmap(throws, title="Spatial density - all simulated player throws"), FIGS / "eda_heatmap_players.png")

    # 4. TRAIN / TEST SPLIT (same split for every model) -----------------------
    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    log(f"Train: {len(train)}  Test: {len(test)}  Classes: {df['label'].nunique()}")

    rows, fitted = [], {}
    # Geometry baseline
    geo = GeometryBaseline().fit(None)
    for split_name, split in [("test", test), ("uniform", uniform)]:
        pred = geo.predict(split[["x", "y"]])
        r = full_report(split["x"], split["y"], split["label"], pred)
        rows.append({"model": "Geometry baseline", "feature_set": "rules", "eval_set": split_name,
                     "trained_on": "rules", "train_seconds": 0.0, **r})
    joblib.dump(geo, MODELS / "geometry_baseline.joblib")

    for fs in FEATURE_SETS:
        for name, model in get_models(seed).items():
            log(f"Training {name} on '{fs}' features ...")
            t0 = time.time()
            model.fit(feature_matrix(train, fs), train["label"])
            secs = time.time() - t0
            fitted[(name, fs)] = (model, round(secs, 2))
            for split_name, split in [("test", test), ("uniform", uniform)]:
                pred = model.predict(feature_matrix(split, fs))
                r = full_report(split["x"], split["y"], split["label"], pred)
                rows.append({"model": name, "feature_set": fs, "eval_set": split_name,
                             "trained_on": "synthetic", "train_seconds": round(secs, 2), **r})
                if split_name == "test":
                    cm = confusion(split["label"], pred)
                    ax = plots.confusion_heatmap(cm, title=f"{name} ({fs} features) - normalised confusion matrix")
                    plots.save(ax, FIGS / f"confusion_{name.lower().replace(' ', '_')}_{fs}.png")
            fname = f"{name.lower().replace(' ', '_')}_{fs}.joblib"
            joblib.dump({"model": model, "feature_set": fs, "name": name}, MODELS / fname)

    # 4b. REAL DATA (DeepDarts, IEEE DataPort) - only if prepare_real_data.py has been run
    real, real_val = realdata.load_real(DATA / "real_throws.csv")
    if real is not None:
        log(f"Real throws found: {len(real)} darts after validation {real_val}")
        rows += realdata.run(real, fitted, seed, test_size, FIGS, RESULTS, log)
    else:
        log("No data/real_throws.csv - skipping real-data experiments (run prepare_real_data.py first)")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    test_metrics = metrics[metrics["eval_set"] == "test"]
    ml_only = test_metrics[test_metrics["model"] != "Geometry baseline"]
    plots.save(plots.model_comparison(ml_only), FIGS / "model_comparison.png")

    # 5. PLAYER ANALYSIS ------------------------------------------------------
    log("Player analysis ...")
    summary = player_summary(throws)
    summary["bias"] = [bias_description(bx, by) for bx, by in zip(summary["bias_x_mm"], summary["bias_y_mm"])]
    summary.to_csv(RESULTS / "player_summary.csv", index=False)
    trend = session_trend(throws)
    trend.to_csv(RESULTS / "player_session_trend.csv", index=False)
    plots.save(plots.landing_map(throws, hue="player", title="Simulated player landing map", s=8), FIGS / "players_landing_map.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    for p, d in trend.groupby("player"):
        ax.plot(d["session"], d["avg_score"], marker="o", label=p)
    ax.set_xlabel("session"); ax.set_ylabel("average score per dart"); ax.set_title("Score trend by session")
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1, 1))
    plots.save(fig, FIGS / "players_session_trend.png")

    # 6. REPORT ---------------------------------------------------------------
    best = ml_only.sort_values("macro_f1", ascending=False).iloc[0]
    json.dump({"best_model": best["model"], "best_feature_set": best["feature_set"],
               "best_model_file": f"{best['model'].lower().replace(' ', '_')}_{best['feature_set']}.joblib"},
              open(MODELS / "best_model.json", "w"), indent=2)
    write_report(metrics, summary, report, best, real is not None)
    log("Done. Scorecard (held-out test set):")
    cols = ["model", "feature_set", "accuracy", "macro_f1", "score_accuracy", "boundary_accuracy"]
    print(test_metrics[cols].round(4).to_string(index=False))
    log(f"Best ML model: {best['model']} ({best['feature_set']}) - macro F1 {best['macro_f1']:.4f}")
    log("Next: streamlit run app.py")


def write_report(metrics, summary, val_report, best, has_real=False):
    t = metrics[metrics["eval_set"] == "test"]
    u = metrics[metrics["eval_set"] == "uniform"]
    cols = ["model", "feature_set", "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "score_accuracy", "score_mae", "boundary_accuracy", "interior_accuracy", "train_seconds"]
    lines = [
        "# AI DartVision - Evaluation Report", "",
        "All numbers below were produced by `run_pipeline.py` on synthetic, geometry-labelled data.",
        "They measure how well each ML model reproduces the deterministic scoring rules - "
        "not real-world player skill.", "",
        "## Data validation", "", f"```\n{val_report}\n```", "",
        "## Scorecard - held-out stratified test set", "", t[cols].round(4).to_markdown(index=False), "",
        "## Scorecard - uniform board coverage (natural class imbalance)", "", u[cols].round(4).to_markdown(index=False), "",
        f"**Best ML model:** {best['model']} with {best['feature_set']} features "
        f"(macro F1 = {best['macro_f1']:.4f}).", "",
        "`boundary_accuracy` = accuracy on throws within ~1.7 mm of a wire; this is where ML models disagree with the rules.", "",
        *(real_section(metrics) if has_real else []),
        "## Simulated player summary", "", summary.to_markdown(index=False), "",
        "## Interpretation", "",
        "* The geometry baseline scores 100% by construction - it is the rule set that produced the labels.",
        "* Engineered polar features (r, sin θ, cos θ) help linear models most, because ring membership is a function of r.",
        "* Remaining ML errors are concentrated at region boundaries (see confusion matrices in `figures/`).",
        "* Player statistics come from simulated throwers; meaningful player analysis requires recorded real throws.",
    ]
    (RESULTS / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def real_section(metrics):
    cols = ["model", "feature_set", "trained_on", "accuracy", "macro_f1", "boundary_accuracy"]
    rs = json.loads((RESULTS / "real_summary.json").read_text())
    tr = metrics[metrics["eval_set"] == "real"][cols].round(4)
    rt = metrics[metrics["eval_set"] == "real_trained"][cols].round(4)
    return [
        "## Real data - DeepDarts (IEEE DataPort)", "",
        f"Source: {rs['citation']}", "",
        f"{rs['darts']} unique real darts in {rs['visits']} visits over {rs['sessions']} sessions "
        f"({rs['date_range'][0]} to {rs['date_range'][1]}). Dart tips were converted to board coordinates with a "
        "homography through the 4 labelled calibration points, then scored with the geometry rules. "
        f"Held-out test: {rs['split']['test_darts']} darts from {rs['split']['test_sessions']} unseen sessions.", "",
        f"Average {rs['avg_score_per_dart']} points per dart, {rs['three_dart_avg']} per 3-dart visit, "
        f"miss rate {rs['miss_rate']:.1%}, treble rate {rs['treble_rate']:.1%}.", "",
        "### Trained on synthetic data, tested on real throws", "", tr.to_markdown(index=False), "",
        "### Trained on real throws, tested on real throws (held-out sessions)", "", rt.to_markdown(index=False), "",
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the AI DartVision pipeline")
    ap.add_argument("--n-per-class", type=int, default=250, help="synthetic points per region (63 regions)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    a = ap.parse_args()
    main(a.n_per_class, a.seed, a.test_size)
    from train_sports import train_all          # archery + shooting targets (Multi-Sport page)
    train_all(seed=a.seed, log=log)