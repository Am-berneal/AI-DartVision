"""
Synthetic data generation.

1. make_synthetic_dataset  -> geometry-labelled coordinates covering every region
   (stratified so that tiny regions like the bull are not under-represented).
2. make_uniform_dataset    -> points spread uniformly over the board area
   (realistic class imbalance, used as an extra test set).
3. simulate_players        -> simulated throwing sessions for player analysis.
   NOTE: these are simulated players, not real people. Real player analysis
   needs recorded throws (see the proposal's key limitation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .board import (LABELS, RADII, SECTORS, SECTOR_WIDTH_DEG, label_to_score,
                    region_center, score_region)

MISS_MAX_R = 1.15  # misses are sampled just outside the board, up to 1.15


def _bands(label: str):
    """Radial bands (r_min, r_max) and clockwise angle window for a label."""
    if label == "IB":
        return [(0.0, RADII["inner_bull"])], None
    if label == "OB":
        return [(RADII["inner_bull"], RADII["outer_bull"])], None
    if label == "MISS":
        return [(RADII["double_outer"], MISS_MAX_R)], None
    kind, num = label[0], int(label[1:])
    centre = SECTORS.index(num) * SECTOR_WIDTH_DEG
    window = (centre - SECTOR_WIDTH_DEG / 2, centre + SECTOR_WIDTH_DEG / 2)
    if kind == "T":
        return [(RADII["triple_inner"], RADII["triple_outer"])], window
    if kind == "D":
        return [(RADII["double_inner"], RADII["double_outer"])], window
    # single: inner single + outer single bands
    return [(RADII["outer_bull"], RADII["triple_inner"]),
            (RADII["triple_outer"], RADII["double_inner"])], window


def sample_region(label: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly (by area) inside one scoring region."""
    bands, window = _bands(label)
    areas = np.array([b[1] ** 2 - b[0] ** 2 for b in bands])
    which = rng.choice(len(bands), size=n, p=areas / areas.sum())
    lo = np.array([bands[i][0] for i in which])
    hi = np.array([bands[i][1] for i in which])
    r = np.sqrt(rng.uniform(lo ** 2, hi ** 2))
    if window is None:
        clock = rng.uniform(0, 360, n)
    else:
        clock = rng.uniform(window[0], window[1], n)
    theta = np.radians(90.0 - clock)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _frame(xy: np.ndarray, source: str) -> pd.DataFrame:
    labels = score_region(xy[:, 0], xy[:, 1])
    return pd.DataFrame({
        "x": xy[:, 0], "y": xy[:, 1], "label": labels,
        "score": [label_to_score(l) for l in labels], "source": source,
    })


def make_synthetic_dataset(n_per_class: int = 250, seed: int = 42) -> pd.DataFrame:
    """Stratified, geometry-labelled dataset with n_per_class points per region."""
    rng = np.random.default_rng(seed)
    xy = np.vstack([sample_region(lbl, n_per_class, rng) for lbl in LABELS])
    df = _frame(xy, "stratified")
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_uniform_dataset(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    """Points uniform over the disc r <= MISS_MAX_R (natural class imbalance)."""
    rng = np.random.default_rng(seed)
    r = MISS_MAX_R * np.sqrt(rng.uniform(0, 1, n))
    t = rng.uniform(0, 2 * np.pi, n)
    return _frame(np.column_stack([r * np.cos(t), r * np.sin(t)]), "uniform")


# Simulated player profiles: aim target, spread (std dev, normalised units), systematic bias (dx, dy)
PLAYER_PROFILES = {
    "Player A (pro)":          {"aim": "T20", "sigma": 0.035, "bias": (0.000, 0.000)},
    "Player B (club)":         {"aim": "T20", "sigma": 0.070, "bias": (0.015, -0.010)},
    "Player C (low & left)":   {"aim": "T20", "sigma": 0.060, "bias": (-0.030, -0.040)},
    "Player D (beginner)":     {"aim": "T20", "sigma": 0.140, "bias": (0.010, -0.050)},
    "Player E (bull hunter)":  {"aim": "IB",  "sigma": 0.060, "bias": (0.005, 0.010)},
}


def simulate_players(n_sessions: int = 5, throws_per_session: int = 60, seed: int = 123,
                     profiles: dict | None = None) -> pd.DataFrame:
    """Simulate throwing sessions. Players improve slightly session over session."""
    rng = np.random.default_rng(seed)
    profiles = profiles or PLAYER_PROFILES
    rows = []
    for player, p in profiles.items():
        ax, ay = region_center(p["aim"])
        for s in range(1, n_sessions + 1):
            improve = 1.0 - 0.04 * (s - 1)  # 4% tighter grouping each session
            sigma = p["sigma"] * improve
            bx, by = p["bias"][0] * improve, p["bias"][1] * improve
            x = ax + bx + rng.normal(0, sigma, throws_per_session)
            y = ay + by + rng.normal(0, sigma, throws_per_session)
            for i in range(throws_per_session):
                rows.append((player, s, i + 1, p["aim"], ax, ay, x[i], y[i]))
    df = pd.DataFrame(rows, columns=["player", "session", "throw", "aim_label",
                                     "aim_x", "aim_y", "x", "y"])
    df["label"] = score_region(df["x"].values, df["y"].values)
    df["score"] = [label_to_score(l) for l in df["label"]]
    return df