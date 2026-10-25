"""
AI DartVision - web API + frontend server (FastAPI).

Run (from the project folder, after `python run_pipeline.py`):
    uvicorn api:app --reload
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

from dartvision import analytics
from dartvision.board import (LABEL_NAMES, RADII, label_to_score, labels_to_scores,
                              polar, score_region, sector_number)
from dartvision.coach import (LANGUAGES, TONES, answer_question, coach_feedback, coach_stats,
                              llm_config, player_facts)
from dartvision.evaluate import full_report
from dartvision.players import bias_description, player_summary, session_trend
from dartvision.preprocessing import add_features, feature_matrix, normalise, validate_coordinates

ROOT = Path(__file__).parent
MODELS, DATA, RESULTS = ROOT / "models", ROOT / "data", ROOT / "results"
FRONTEND = ROOT / "frontend"


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into the environment (keeps the API key out of the code)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(ROOT / ".env")

app = FastAPI(title="AI DartVision API")


# ------------------------------------------------------------------ helpers
def model_keys() -> list[str]:
    return sorted(p.stem for p in MODELS.glob("*.joblib") if p.stem != "geometry_baseline")


@lru_cache(maxsize=16)
def load_model(key: str) -> dict:
    path = MODELS / f"{key}.joblib"
    if not path.exists():
        raise HTTPException(404, f"Model '{key}' not found. Run: python run_pipeline.py")
    return joblib.load(path)


def best_key() -> str | None:
    p = MODELS / "best_model.json"
    if p.exists():
        return json.loads(p.read_text())["best_model_file"].replace(".joblib", "")
    keys = model_keys()
    return keys[0] if keys else None


def resolve(key: str | None) -> tuple[str, dict]:
    key = key or best_key()
    if not key:
        raise HTTPException(503, "No trained models found. Run: python run_pipeline.py")
    return key, load_model(key)


@lru_cache(maxsize=1)
def throws_df() -> pd.DataFrame:
    p = DATA / "player_throws.csv"
    if not p.exists():
        raise HTTPException(503, "data/player_throws.csv missing. Run: python run_pipeline.py")
    return pd.read_csv(p)


@lru_cache(maxsize=1)
def metrics_df() -> pd.DataFrame:
    p = RESULTS / "metrics.csv"
    if not p.exists():
        raise HTTPException(503, "results/metrics.csv missing. Run: python run_pipeline.py")
    return pd.read_csv(p)


@lru_cache(maxsize=1)
def test_split() -> pd.DataFrame:
    # Same split as run_pipeline.py (default seed 42, test_size 0.2, stratified)
    df = pd.read_csv(DATA / "synthetic_coordinates.csv")
    _, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    return test


def describe(label: str) -> dict:
    if label in ("IB", "OB", "MISS"):
        return {"ring": LABEL_NAMES[label], "multiplier": 1, "base": label_to_score(label)}
    kind = {"S": "Single", "D": "Double", "T": "Triple"}[label[0]]
    return {"ring": kind, "multiplier": {"S": 1, "D": 2, "T": 3}[label[0]], "base": int(label[1:])}


def to_norm(x, y, unit: str):
    return normalise(x, y, unit="mm" if unit == "mm" else "norm")


def clean(obj):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ------------------------------------------------------------------ schemas
class PredictIn(BaseModel):
    x: float
    y: float
    unit: str = "norm"
    model: str | None = None


class BatchRow(BaseModel):
    x: float
    y: float
    player: str | None = None
    target: str | None = None
    session: str | int | None = None


class BatchIn(BaseModel):
    rows: list[BatchRow]
    unit: str = "norm"
    model: str | None = None


class CoachIn(BaseModel):
    player: str
    question: str | None = None
    language: str = "English"
    tone: str = "friendly"


# ------------------------------------------------------------------ routes
@app.get("/api/status")
def status():
    keys = model_keys()
    return {"ok": True, "models": len(keys), "best": best_key(),
            "pipeline_run": (RESULTS / "metrics.csv").exists()}


@app.get("/api/models")
def models():
    out = []
    for k in model_keys():
        name, fs = k.rsplit("_", 1)
        out.append({"key": k, "name": name.replace("_", " ").title().replace("Knn", "KNN").replace("Svm", "SVM"),
                    "feature_set": fs})
    return {"models": out, "best": best_key()}


@app.post("/api/predict")
def predict(p: PredictIn):
    key, m = resolve(p.model)
    nx, ny = to_norm([p.x], [p.y], p.unit)
    x, y = float(nx[0]), float(ny[0])
    X = feature_matrix(add_features(pd.DataFrame({"x": [x], "y": [y]})), m["feature_set"])
    t0 = time.perf_counter()
    label = str(m["model"].predict(X)[0])
    infer_ms = (time.perf_counter() - t0) * 1000
    conf = None
    if hasattr(m["model"], "predict_proba"):
        try:
            conf = float(np.max(m["model"].predict_proba(X)[0]))
        except Exception:  # noqa: BLE001
            conf = None
    geo = str(score_region([x], [y])[0])
    r, clock = polar([x], [y])
    return clean({
        "model": key, "model_name": m["name"], "feature_set": m["feature_set"],
        "x": x, "y": y, "r": float(r[0]), "theta_deg": float(np.degrees(np.arctan2(y, x))),
        "x_mm": x * 170, "y_mm": y * 170, "r_mm": float(r[0]) * 170,
        "label": label, "label_name": LABEL_NAMES.get(label, label), "score": label_to_score(label),
        "geometry_label": geo, "geometry_score": label_to_score(geo), "agrees": label == geo,
        "confidence": conf, "inference_ms": infer_ms, "sector": int(sector_number([x], [y])[0]),
        **describe(label),
    })


@app.post("/api/batch")
def batch(b: BatchIn):
    if not b.rows:
        raise HTTPException(400, "No rows")
    key, m = resolve(b.model)
    raw_x = np.array([r.x for r in b.rows])
    raw_y = np.array([r.y for r in b.rows])
    nx, ny = to_norm(raw_x, raw_y, b.unit)
    work = add_features(pd.DataFrame({"x": nx, "y": ny}))
    pred = m["model"].predict(feature_matrix(work, m["feature_set"])).astype(str)
    geo = score_region(nx, ny)
    scores = labels_to_scores(pred)
    rows = []
    for i, r in enumerate(b.rows):
        rows.append({"n": i + 1, "player": r.player or "", "x": float(raw_x[i]), "y": float(raw_y[i]),
                     "x_norm": float(nx[i]), "y_norm": float(ny[i]),
                     "predicted": pred[i], "predicted_name": LABEL_NAMES.get(pred[i], pred[i]),
                     "score": int(scores[i]), "geometry": geo[i], "agrees": bool(pred[i] == geo[i])})
    rr = work["r"].values
    peak = int(np.argmax(scores))
    # Per-player summary of the uploaded throws (extra field; the fields above are unchanged)
    try:
        summ_df = pd.DataFrame({"x": nx, "y": ny, "label": geo, "score": labels_to_scores(geo),
                                "player": [r.player or "" for r in b.rows],
                                "target": [r.target or "" for r in b.rows],
                                "session": ["" if r.session is None else str(r.session) for r in b.rows]})
        player_summary_out = analytics.batch_summary(summ_df)
    except Exception as e:  # noqa: BLE001 - the summary must never break batch scoring
        player_summary_out = {"players": [], "notes": [f"Summary unavailable: {type(e).__name__}"], "error": True}
    return clean({
        "model": key, "model_name": m["name"], "feature_set": m["feature_set"], "rows": rows,
        "player_summary": player_summary_out,
        "summary": {
            "total": int(scores.sum()), "n": len(rows), "mean_score": float(scores.mean()),
            "peak_score": int(scores[peak]), "peak_label": pred[peak],
            "bulls": int(np.isin(pred, ["IB", "OB"]).sum()),
            "misses": int((pred == "MISS").sum()),
            "agreement": float(np.mean(pred == geo)), "discrepancies": int(np.sum(pred != geo)),
            "mean_r": float(rr.mean()), "std_r": float(rr.std()),
            "r_min": float(rr.min()), "r_max": float(rr.max()),
        },
    })


@app.get("/api/sample")
def sample():
    p = DATA / "sample_throws.csv"
    if not p.exists():
        raise HTTPException(404, "data/sample_throws.csv not found")
    return {"rows": pd.read_csv(p).to_dict(orient="records"), "file": p.name}


@app.get("/api/players")
def players():
    t = throws_df()
    s = player_summary(t)
    s["bias"] = [bias_description(bx, by) for bx, by in zip(s["bias_x_mm"], s["bias_y_mm"])]
    tr = session_trend(t)
    form = {}
    for p, d in tr.groupby("player"):
        d = d.sort_values("session")
        delta = d["avg_score"].iloc[-1] - d["avg_score"].iloc[0]
        form[p] = "RISING" if delta > 1.5 else ("FALLING" if delta < -1.5 else "STABLE")
    s["form"] = s["player"].map(form)
    return clean({"players": s.to_dict(orient="records")})


@app.get("/api/players/{name}")
def player(name: str):
    t = throws_df()
    pt = t[t["player"] == name]
    if pt.empty:
        raise HTTPException(404, f"Unknown player {name}")
    facts = player_facts(t, name)
    tr = session_trend(t)
    tr = tr[tr["player"] == name].sort_values("session")
    tr = tr.assign(three_dart_avg=tr["avg_score"] * 3)
    cohort = session_trend(t).groupby("player")["avg_score"].mean().mean() * 3
    return clean({
        "facts": facts,
        "aim": {"x": float(pt["aim_x"].iloc[0]), "y": float(pt["aim_y"].iloc[0])},
        "throws": pt[["session", "throw", "x", "y", "label", "score"]].to_dict(orient="records"),
        "trend": tr.to_dict(orient="records"),
        "cohort_3da": float(cohort),
    })


@lru_cache(maxsize=1)
def _roster_profile() -> dict:
    return analytics.roster_profile(throws_df())


@lru_cache(maxsize=16)
def _player_analytics_cached(name: str) -> dict:
    return clean(analytics.player_analytics(throws_df(), name, _roster_profile()))


@app.get("/api/players/{name}/analytics")
def player_analytics(name: str):
    try:
        return _player_analytics_cached(name)
    except KeyError:
        raise HTTPException(404, f"Unknown player {name}")


@app.get("/api/compare")
def compare(a: str, b: str):
    try:
        return clean(analytics.compare_players(throws_df(), a, b))
    except KeyError as e:
        raise HTTPException(404, f"Unknown player {e.args[0]}")


@lru_cache(maxsize=1)
def real_df() -> pd.DataFrame:
    p = DATA / "real_throws.csv"
    if not p.exists():
        raise HTTPException(404, "No real data yet. Run python prepare_real_data.py and python run_pipeline.py")
    df, _ = validate_coordinates(pd.read_csv(p))      # same cleaning as the pipeline (drops duplicates / out of range)
    df["date"] = pd.to_datetime(df["session"].str.slice(3, 13), format="%m_%d_%Y", errors="coerce")
    return df


@lru_cache(maxsize=4)
def _real_analytics_cached(subset: str) -> dict:
    return clean(analytics.real_analytics(real_df(), None if subset == "all" else subset))


@app.get("/api/real/analytics")
def real_analytics(subset: str = "all"):
    if subset not in ("all", "D1", "D2"):
        raise HTTPException(400, "subset must be all, D1 or D2")
    return _real_analytics_cached(subset)


@app.get("/api/coach/status")
def coach_status():
    cfg = llm_config()
    return {**cfg, "languages": list(LANGUAGES), "tones": list(TONES), "stats": coach_stats()}


@app.post("/api/coach")
def coach(c: CoachIn):
    t = throws_df()
    try:
        if c.question:
            return answer_question(t, c.player, c.question, c.language)
        return clean(coach_feedback(t, c.player, c.language, c.tone))
    except KeyError:
        raise HTTPException(404, f"Unknown player {c.player}")


@app.get("/api/real")
def real():
    p = RESULTS / "real_summary.json"
    if not p.exists():
        raise HTTPException(404, "No real data yet. Put labels.pkl in data/deepdarts/, then run "
                                 "python prepare_real_data.py and python run_pipeline.py")
    summary = json.loads(p.read_text(encoding="utf-8"))
    m = metrics_df()
    rows = m[m["eval_set"].isin(["real", "real_trained"])]
    return clean({**summary, "results": rows.to_dict(orient="records")})


@app.get("/api/metrics")
def metrics():
    return clean({"rows": metrics_df().to_dict(orient="records")})


KEY_LABELS = ["T20", "S20", "D20", "T1", "S1", "T5", "S5", "T19", "S19", "OB", "IB", "MISS"]


@app.get("/api/confusion")
def confusion(model: str | None = None):
    key, m = resolve(model)
    test = test_split()
    pred = m["model"].predict(feature_matrix(test, m["feature_set"])).astype(str)
    true = test["label"].values.astype(str)
    cols = KEY_LABELS + ["Other"]
    matrix = []
    for lbl in KEY_LABELS:
        mask = true == lbl
        n = mask.sum()
        row = []
        for c in KEY_LABELS:
            row.append(float(np.sum(pred[mask] == c) / n * 100) if n else 0.0)
        row.append(float(np.sum(~np.isin(pred[mask], KEY_LABELS)) / n * 100) if n else 0.0)
        matrix.append(row)
    rep = full_report(test["x"].values, test["y"].values, true, pred)
    trebles = sorted({l for l in true if l.startswith("T")})
    rep["treble_precision"] = precision_score(true, pred, labels=trebles, average="macro", zero_division=0)
    return clean({"model": key, "model_name": m["name"], "feature_set": m["feature_set"],
                  "rows": KEY_LABELS, "cols": cols, "matrix": matrix, "report": rep, "n_test": int(len(test))})


# Frontend (must be mounted last so /api routes take priority)
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")