"""
Preprocessing and feature engineering.

* validate_coordinates : drop missing / non-numeric / absurd coordinates
* normalise            : convert mm or pixel coordinates to board-normalised units,
                         correct orientation (rotation) and flip the y-axis if needed
* add_features         : r, sin(theta), cos(theta)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .board import BOARD_RADIUS_MM

MAX_VALID_R = 1.5  # anything further than 1.5 board radii is treated as bad data

FEATURE_SETS = {
    "raw": ["x", "y"],
    "engineered": ["x", "y", "r", "sin_t", "cos_t"],
}


def validate_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a cleaned copy plus a small report of what was removed."""
    out = df.copy()
    report = {"rows_in": len(out)}
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    finite = np.isfinite(out["x"]) & np.isfinite(out["y"])
    report["dropped_non_numeric"] = int((~finite).sum())
    out = out[finite]
    in_range = np.hypot(out["x"], out["y"]) <= MAX_VALID_R
    report["dropped_out_of_range"] = int((~in_range).sum())
    out = out[in_range]
    before = len(out)
    out = out.drop_duplicates(subset=["x", "y"])
    report["dropped_duplicates"] = before - len(out)
    report["rows_out"] = len(out)
    return out.reset_index(drop=True), report


def normalise(x, y, unit: str = "norm", board_radius_px: float | None = None,
              centre_px: tuple[float, float] | None = None, rotation_deg: float = 0.0,
              flip_y: bool = False):
    """Convert raw coordinates into board-normalised coordinates.

    unit: 'norm' (already normalised), 'mm' (board-centred millimetres) or
          'px' (image pixels; needs board_radius_px and centre_px).
    rotation_deg: rotate counter-clockwise to correct a tilted board/camera.
    flip_y: image coordinates have y pointing down; set True to fix that.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if unit == "mm":
        x, y = x / BOARD_RADIUS_MM, y / BOARD_RADIUS_MM
    elif unit == "px":
        if board_radius_px is None or centre_px is None:
            raise ValueError("pixel input needs board_radius_px and centre_px")
        x = (x - centre_px[0]) / board_radius_px
        y = (y - centre_px[1]) / board_radius_px
    elif unit != "norm":
        raise ValueError(f"unknown unit {unit!r}")
    if flip_y:
        y = -y
    if rotation_deg:
        a = np.radians(rotation_deg)
        x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    return x, y


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["r"] = np.hypot(out["x"], out["y"])
    theta = np.arctan2(out["y"], out["x"])
    out["sin_t"] = np.sin(theta)
    out["cos_t"] = np.cos(theta)
    return out


def feature_matrix(df: pd.DataFrame, feature_set: str = "engineered") -> pd.DataFrame:
    cols = FEATURE_SETS[feature_set]
    if any(c not in df.columns for c in cols):
        df = add_features(df)
    return df[cols]