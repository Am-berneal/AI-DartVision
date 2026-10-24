"""
Real throws from the DeepDarts dataset (IEEE DataPort, DOI 10.21227/05e7-xs69).

McNally et al., "DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts
using a Single Camera", CVPR Sports Workshop 2021. 16,050 photos, each labelled with:
  * 4 calibration points on the outer edge of the double ring, at the wire boundaries
    20/5 (top), 3/17 (bottom), 8/11 (left) and 13/6 (right)
  * up to 3 dart-tip positions
Points are stored as fractions (0-1) of a square crop of the photo, with y pointing down.

This module converts every dart tip into the project's board coordinates
(centre (0,0), y up, double-ring outer edge r = 1) with a perspective transform (homography)
fitted through the 4 calibration points, then scores it with the geometry rules.
"""
from __future__ import annotations

import importlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .board import label_to_score, score_region

# Clockwise angle from the top (degrees) of each calibration point, in label order.
CAL_ANGLES = [-9.0, 171.0, 261.0, 81.0]          # 20/5, 3/17, 8/11, 13/6 wires
CAL_BOARD = np.array([[np.sin(np.radians(a)), np.cos(np.radians(a))] for a in CAL_ANGLES])


class _SafeUnpickler(pickle.Unpickler):
    """Loads the labels file but only allows pandas / numpy objects (pickle can run code)."""

    SAFE_BUILTINS = {"slice", "list", "dict", "tuple", "set", "str", "int", "float"}

    def find_class(self, module, name):
        root = module.split(".")[0]
        if root not in ("pandas", "numpy", "builtins"):
            raise pickle.UnpicklingError(f"Blocked unsafe object {module}.{name}")
        if root == "builtins" and name not in self.SAFE_BUILTINS:
            raise pickle.UnpicklingError(f"Blocked unsafe builtin {name}")
        if module.startswith("numpy.core"):          # files saved with older numpy
            try:
                return getattr(importlib.import_module(module.replace("numpy.core", "numpy._core", 1)), name)
            except (ImportError, AttributeError):
                pass
        return getattr(importlib.import_module(module), name)


def load_labels(path: str | Path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        df = _SafeUnpickler(fh).load()
    if not isinstance(df, pd.DataFrame) or not {"img_folder", "img_name", "xy"} <= set(df.columns):
        raise ValueError("This does not look like the DeepDarts labels.pkl file")
    return df


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 perspective transform mapping 4 src points onto 4 dst points (DLT, h33 = 1)."""
    A, b = [], []
    for (u, v), (X, Y) in zip(src, dst):
        A.append([u, v, 1, 0, 0, 0, -u * X, -v * X]); b.append(X)
        A.append([0, 0, 0, u, v, 1, -u * Y, -v * Y]); b.append(Y)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def to_board(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per unique dart, in board coordinates, scored with the geometry rules."""
    rows = []
    labels = labels.sort_values(["img_folder", "img_name"]).reset_index(drop=True)
    for folder, grp in labels.groupby("img_folder", sort=False):
        seen, visit, prev_first = set(), 0, None
        for r in grp.itertuples(index=False):
            pts = np.asarray(r.xy, dtype=float)
            if len(pts) < 5:            # calibration only, no dart in the photo
                continue
            try:
                H = homography(pts[:4], CAL_BOARD)
            except np.linalg.LinAlgError:
                continue
            darts = apply_h(H, pts[4:])
            first = tuple(np.round(darts[0], 3))
            if first != prev_first:     # a new visit (set of up to 3 darts) starts
                visit += 1
                prev_first = first
            for i, (x, y) in enumerate(darts, start=1):
                key = (visit, i)
                if key in seen:         # the same dart appears again in the next photo
                    continue
                seen.add(key)
                rows.append((folder, r.img_name, visit, i, x, y))
    out = pd.DataFrame(rows, columns=["session", "img_name", "visit", "dart", "x", "y"])
    if out.empty:
        return out.assign(subset=[], visit_id=[], r=[], label=[], score=[])
    out["subset"] = out["session"].str[:2].str.upper()          # D1 (face-on) / D2 (angled)
    out["visit_id"] = out["session"] + "#" + out["visit"].astype(str)
    out["r"] = np.hypot(out["x"], out["y"])
    out["label"] = score_region(out["x"].values, out["y"].values)
    out["score"] = [label_to_score(l) for l in out["label"]]
    return out


def calibration_check(labels: pd.DataFrame, n: int = 200) -> float:
    """Max error (normalised units) when mapping the calibration points themselves - should be ~0."""
    errs = []
    for r in labels.head(n).itertuples(index=False):
        pts = np.asarray(r.xy, dtype=float)
        H = homography(pts[:4], CAL_BOARD)
        errs.append(np.abs(apply_h(H, pts[:4]) - CAL_BOARD).max())
    return float(np.max(errs))