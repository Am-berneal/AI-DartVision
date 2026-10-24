"""Player-level accuracy summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def player_summary(throws: pd.DataFrame) -> pd.DataFrame:
    t = throws.copy()
    t["dx"] = t["x"] - t["aim_x"]
    t["dy"] = t["y"] - t["aim_y"]
    t["miss_dist"] = np.hypot(t["dx"], t["dy"])
    t["hit_target"] = t["label"] == t["aim_label"]
    g = t.groupby("player")
    out = pd.DataFrame({
        "aim": g["aim_label"].first(),
        "throws": g.size(),
        "hit_rate_target": g["hit_target"].mean(),
        "avg_score": g["score"].mean(),
        "three_dart_avg": g["score"].mean() * 3,
        "mean_error_mm": g["miss_dist"].mean() * 170,
        "spread_x_mm": g["dx"].std() * 170,
        "spread_y_mm": g["dy"].std() * 170,
        "bias_x_mm": g["dx"].mean() * 170,
        "bias_y_mm": g["dy"].mean() * 170,
        "miss_rate": g["label"].apply(lambda s: (s == "MISS").mean()),
    })
    return out.round(3).reset_index()


def session_trend(throws: pd.DataFrame) -> pd.DataFrame:
    t = throws.copy()
    t["hit_target"] = t["label"] == t["aim_label"]
    t["miss_dist_mm"] = np.hypot(t["x"] - t["aim_x"], t["y"] - t["aim_y"]) * 170
    return (t.groupby(["player", "session"])
             .agg(avg_score=("score", "mean"), hit_rate=("hit_target", "mean"),
                  mean_error_mm=("miss_dist_mm", "mean"))
             .round(3).reset_index())


def bias_description(bias_x_mm: float, bias_y_mm: float, tol: float = 2.0) -> str:
    parts = []
    if bias_y_mm < -tol:
        parts.append("low")
    elif bias_y_mm > tol:
        parts.append("high")
    if bias_x_mm < -tol:
        parts.append("left")
    elif bias_x_mm > tol:
        parts.append("right")
    return " & ".join(parts) if parts else "centred"