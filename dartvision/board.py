"""
Dartboard geometry and the deterministic geometry-based scorer (reference baseline).

Coordinate convention (used everywhere in the project):
  * Board centre is (0, 0).
  * x points right, y points up (the 20 sector is straight up).
  * Coordinates are NORMALISED so that the outer edge of the double ring is r = 1.0.
    (1.0 normalised unit = 170 mm on a standard board.)

Board dimensions follow the standard WDF / BDO board (in mm, measured to the
inside of the wire / centre of the band):
  inner bull radius 6.35, outer bull 15.9, triple ring 99-107, double ring 162-170.
"""
from __future__ import annotations

import numpy as np

# Sector numbers clockwise starting from the top (12 o'clock).
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
SECTOR_WIDTH_DEG = 360.0 / len(SECTORS)  # 18 degrees

BOARD_RADIUS_MM = 170.0
RADII_MM = {
    "inner_bull": 6.35,
    "outer_bull": 15.9,
    "triple_inner": 99.0,
    "triple_outer": 107.0,
    "double_inner": 162.0,
    "double_outer": 170.0,
}
RADII = {k: v / BOARD_RADIUS_MM for k, v in RADII_MM.items()}
RING_EDGES = sorted(RADII.values())

# Full label set: inner bull, outer bull, 20 singles, 20 doubles, 20 trebles, miss = 63 classes
LABELS = (
    ["IB", "OB"]
    + [f"S{n}" for n in range(1, 21)]
    + [f"D{n}" for n in range(1, 21)]
    + [f"T{n}" for n in range(1, 21)]
    + ["MISS"]
)

LABEL_NAMES = {"IB": "Inner bull", "OB": "Outer bull", "MISS": "Miss"}
for _n in range(1, 21):
    LABEL_NAMES[f"S{_n}"] = f"Single {_n}"
    LABEL_NAMES[f"D{_n}"] = f"Double {_n}"
    LABEL_NAMES[f"T{_n}"] = f"Triple {_n}"


def label_to_score(label: str) -> int:
    """Rule-based score for a region label (e.g. 'T20' -> 60)."""
    if label == "IB":
        return 50
    if label == "OB":
        return 25
    if label == "MISS":
        return 0
    mult = {"S": 1, "D": 2, "T": 3}[label[0]]
    return mult * int(label[1:])


def labels_to_scores(labels) -> np.ndarray:
    return np.array([label_to_score(l) for l in labels], dtype=int)


def polar(x, y):
    """Return radius and clockwise angle from the top (degrees, 0-360)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    theta = np.degrees(np.arctan2(y, x))           # standard maths angle
    clock = np.mod(90.0 - theta, 360.0)             # 0 = top, increases clockwise
    return r, clock


def sector_number(x, y) -> np.ndarray:
    _, clock = polar(x, y)
    idx = np.floor(np.mod(clock + SECTOR_WIDTH_DEG / 2, 360.0) / SECTOR_WIDTH_DEG).astype(int) % 20
    return np.array(SECTORS)[idx]


def score_region(x, y) -> np.ndarray:
    """Deterministic geometry scorer: coordinates -> region labels (vectorised)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    r, _ = polar(x, y)
    num = sector_number(x, y).astype(str)

    out = np.empty(r.shape, dtype=object)
    out[:] = np.char.add("S", num)
    in_triple = (r >= RADII["triple_inner"]) & (r < RADII["triple_outer"])
    in_double = (r >= RADII["double_inner"]) & (r < RADII["double_outer"])
    out[in_triple] = np.char.add("T", num[in_triple])
    out[in_double] = np.char.add("D", num[in_double])
    out[r < RADII["outer_bull"]] = "OB"
    out[r < RADII["inner_bull"]] = "IB"
    out[r >= RADII["double_outer"]] = "MISS"
    return out.astype(str)


def score_point(x: float, y: float) -> tuple[str, int]:
    label = str(score_region([x], [y])[0])
    return label, label_to_score(label)


def distance_to_boundary(x, y) -> np.ndarray:
    """Approximate distance (normalised units) from each point to the nearest region edge.

    Used for 'boundary checks': ML models are expected to struggle near wires.
    """
    r, clock = polar(x, y)
    radial = np.min(np.abs(r[:, None] - np.array(RING_EDGES)[None, :]), axis=1)
    # angular distance to nearest sector wire, converted to arc length
    offset = np.mod(clock + SECTOR_WIDTH_DEG / 2, SECTOR_WIDTH_DEG)
    ang = np.minimum(offset, SECTOR_WIDTH_DEG - offset)
    arc = r * np.radians(ang)
    # sector wires only exist between outer bull and the double ring
    has_wires = (r >= RADII["outer_bull"]) & (r < RADII["double_outer"])
    arc = np.where(has_wires, arc, np.inf)
    return np.minimum(radial, arc)


def region_center(label: str) -> tuple[float, float]:
    """A representative aiming point for a label (centre of the region)."""
    if label in ("IB", "OB"):
        return (0.0, 0.0) if label == "IB" else (0.0, (RADII["inner_bull"] + RADII["outer_bull"]) / 2)
    if label == "MISS":
        return (0.0, 1.1)
    kind, num = label[0], int(label[1:])
    r = {
        "T": (RADII["triple_inner"] + RADII["triple_outer"]) / 2,
        "D": (RADII["double_inner"] + RADII["double_outer"]) / 2,
        "S": (RADII["triple_outer"] + RADII["double_inner"]) / 2,  # big outer single
    }[kind]
    clock = SECTORS.index(num) * SECTOR_WIDTH_DEG
    theta = np.radians(90.0 - clock)
    return float(r * np.cos(theta)), float(r * np.sin(theta))