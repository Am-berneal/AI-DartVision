"""
Multi-sport target scoring: archery (World Archery 10-zone faces) and ISSF 10 m shooting.

All coordinates are millimetres from the target centre (x right, y up).
Every target here is a set of concentric rings, so the score depends only on the distance d from the centre:

* The projectile has a size. In archery an arrow touching a dividing line scores the higher zone
  ("line cutter"); in shooting the ring touched by the pellet's edge counts. So a shot scores ring k
  when (d - projectile_radius) <= outer radius of ring k.
* Shooting also has decimal scoring: each ring width is split into tenths, up to 10.9.

Target specifications are kept in one place (PROFILES) so they can be checked against the rulebooks.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetProfile:
    key: str
    name: str
    sport: str                   # "archery" | "shooting"
    distance_m: float            # shooting distance
    ring_width_mm: float         # radial width of each scoring ring
    ten_radius_mm: float         # outer radius of the 10 ring (printed)
    x_radius_mm: float | None    # inner 10 / X ring radius (printed), None if not used
    projectile_mm: float         # arrow shaft or pellet diameter
    decimal: bool                # ISSF decimal scoring (max 10.9)
    face_mm: float               # printed face size (for drawing)
    colours: dict                # ring value -> fill colour
    source: str                  # where the numbers come from
    notes: str = ""

    @property
    def proj_r(self) -> float:
        return self.projectile_mm / 2

    def ring_radius(self, k: int) -> float:
        """Printed outer radius of ring k (10 = innermost)."""
        return self.ten_radius_mm + (10 - k) * self.ring_width_mm

    @property
    def outer_mm(self) -> float:          # printed outer edge of ring 1
        return self.ring_radius(1)

    @property
    def scoring_limit_mm(self) -> float:  # centre distance beyond which the shot misses
        return self.outer_mm + self.proj_r

    @property
    def labels(self) -> list[str]:
        return (["X"] if self.x_radius_mm else []) + [str(k) for k in range(10, 0, -1)] + ["M"]


ARCHERY_COLOURS = {10: "#F2C14E", 9: "#F2C14E", 8: "#D9443A", 7: "#D9443A", 6: "#3D8FD1", 5: "#3D8FD1",
                   4: "#1E2433", 3: "#1E2433", 2: "#F7F3EA", 1: "#F7F3EA"}
RIFLE_COLOURS = {k: ("#1E2433" if k >= 4 else "#F7F3EA") for k in range(1, 11)}    # black aiming mark = rings 4-10
PISTOL_COLOURS = {k: ("#1E2433" if k >= 7 else "#F7F3EA") for k in range(1, 11)}   # black aiming mark = rings 7-10

ARROW_MM = 5.5   # typical outdoor recurve arrow shaft; change here if your arrows differ
PELLET_MM = 4.5  # 4.5 mm (.177) air gun pellet

PROFILES: dict[str, TargetProfile] = {p.key: p for p in [
    TargetProfile("archery_122", "Archery · 122 cm face (70 m)", "archery", 70, 61.0, 61.0, 30.5, ARROW_MM, False, 1220,
                  ARCHERY_COLOURS, "World Archery 10-zone face: 122 cm, 10 equal rings, inner 10 (X)",
                  "Outdoor recurve Olympic distance. Arrow touching a line scores the higher zone. X counts 10 (tie-breaker)."),
    TargetProfile("archery_40", "Archery · 40 cm face (18 m indoor)", "archery", 18, 20.0, 20.0, 10.0, ARROW_MM, False, 400,
                  ARCHERY_COLOURS, "World Archery 10-zone face: 40 cm, 10 equal rings, inner 10 (X)",
                  "Indoor face. Arrow touching a line scores the higher zone."),
    TargetProfile("air_rifle", "Shooting · 10 m air rifle", "shooting", 10, 2.5, 0.25, None, PELLET_MM, True, 45.5,
                  RIFLE_COLOURS, "ISSF 10 m air rifle: 10 ring 0.5 mm, rings 5 mm apart in diameter, ring 1 = 45.5 mm",
                  "Ring touched by the pellet edge counts. Decimal scoring up to 10.9."),
    TargetProfile("air_pistol", "Shooting · 10 m air pistol", "shooting", 10, 8.0, 5.75, 2.5, PELLET_MM, True, 155.5,
                  PISTOL_COLOURS, "ISSF 10 m air pistol: 10 ring 11.5 mm, inner ten 5 mm, rings 16 mm apart in diameter, ring 1 = 155.5 mm",
                  "Ring touched by the pellet edge counts. Decimal scoring up to 10.9; inner ten is a tie-breaker."),
]}

# Reference targets used only in the cross-sport difficulty table (not trained)
REFERENCE_TARGETS = [
    # name, sport, effective radius of the top zone (mm), distance (m), note
    ("Darts · inner bull (50)", "darts", 6.35, 2.37, "WDF board, oche 2.37 m"),
    ("Darts · treble ring (half width)", "darts", 4.0, 2.37, "treble ring is 8 mm wide"),
    ("Biathlon · prone target", "biathlon", 22.5, 50, "45 mm hit zone at 50 m"),
    ("Biathlon · standing target", "biathlon", 57.5, 50, "115 mm hit zone at 50 m"),
]


def get_profile(key: str) -> TargetProfile:
    if key not in PROFILES:
        raise KeyError(key)
    return PROFILES[key]


# ------------------------------------------------------------------ scoring
def score_shots(p: TargetProfile, x, y) -> pd.DataFrame:
    """Score shots (mm from centre). Returns label, ring value, decimal (shooting only) and distances."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.hypot(x, y)
    w = p.ring_width_mm
    if p.decimal:
        # tenths of a ring: 10.0 exactly when the pellet edge touches the 10 ring, 10.9 at dead centre
        raw = 11.0 - (d - p.proj_r - p.ten_radius_mm + w) / w
        dec = np.floor(np.round(raw * 10, 6)) / 10
        dec = np.clip(dec, None, 10.9)
        dec = np.where(dec < 1.0, 0.0, dec)
        ring = np.floor(dec).astype(int)
    else:
        e = np.maximum(d - p.proj_r - p.ten_radius_mm, 0.0)     # distance of the arrow edge beyond the 10 line
        ring = 10 - np.ceil(np.round(e / w, 9)).astype(int)
        ring = np.where(ring < 1, 0, ring)
        dec = np.full(len(d), np.nan)
    labels = np.where(ring == 0, "M", ring.astype(str)).astype(object)
    if p.x_radius_mm:
        labels = np.where((ring == 10) & (d - p.proj_r <= p.x_radius_mm), "X", labels)
    return pd.DataFrame({"x": x, "y": y, "d_mm": d, "label": labels, "value": ring,
                         "decimal": dec if p.decimal else np.full(len(d), np.nan)})


def boundary_distances(p: TargetProfile, d) -> np.ndarray:
    """Distance (mm) from each shot centre to the nearest effective scoring line (incl. projectile size)."""
    d = np.asarray(d, dtype=float)[:, None]
    lines = [p.ring_radius(k) + p.proj_r for k in range(10, 0, -1)]
    if p.x_radius_mm:
        lines.append(p.x_radius_mm + p.proj_r)
    return np.min(np.abs(d - np.array(lines)[None, :]), axis=1)


# ------------------------------------------------------------------ synthetic data (for the ML models)
def _zone_bounds(p: TargetProfile, label: str) -> tuple[float, float]:
    """Centre-distance interval (mm) that produces this label."""
    r = p.proj_r
    if label == "X":
        return 0.0, p.x_radius_mm + r
    if label == "M":
        return p.scoring_limit_mm, p.scoring_limit_mm + 2.5 * p.ring_width_mm
    k = int(label)
    inner = 0.0 if k == 10 else p.ring_radius(k + 1) + r
    if k == 10 and p.x_radius_mm:
        inner = p.x_radius_mm + r
    return inner, p.ring_radius(k) + r


def make_dataset(p: TargetProfile, n_per_class: int = 300, seed: int = 0) -> pd.DataFrame:
    """Stratified synthetic shots: the same number per zone, area-uniform inside each zone, labelled by the rules."""
    rng = np.random.default_rng(seed)
    parts = []
    for lab in p.labels:
        lo, hi = _zone_bounds(p, lab)
        rad = np.sqrt(rng.uniform(lo ** 2, hi ** 2, n_per_class))
        ang = rng.uniform(0, 2 * np.pi, n_per_class)
        parts.append(pd.DataFrame({"x": rad * np.cos(ang), "y": rad * np.sin(ang)}))
    df = pd.concat(parts, ignore_index=True)
    s = score_shots(p, df["x"], df["y"])
    df["label"] = s["label"].astype(str)
    return df


def normalised(p: TargetProfile, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Scale mm to target units: 1.0 = centre distance at which shots stop scoring."""
    return np.asarray(x, float) / p.scoring_limit_mm, np.asarray(y, float) / p.scoring_limit_mm


# ------------------------------------------------------------------ simulation and group statistics
def simulate(p: TargetProfile, n: int = 60, sigma_mrad: float = 1.0, bias_x_mm: float = 0.0,
             bias_y_mm: float = 0.0, seed: int | None = None) -> pd.DataFrame:
    """Simulated shooter: circular normal spread. sigma (mm per axis) = sigma_mrad x distance (m)."""
    rng = np.random.default_rng(seed)
    sigma_mm = sigma_mrad * p.distance_m
    x = rng.normal(bias_x_mm, sigma_mm, n)
    y = rng.normal(bias_y_mm, sigma_mm, n)
    return score_shots(p, x, y)


def group_stats(p: TargetProfile, scored: pd.DataFrame) -> dict:
    n = len(scored)
    if n == 0:
        return {"shots": 0}
    total = float(scored["decimal"].sum()) if p.decimal else float(scored["value"].sum())
    out = {
        "shots": n,
        "total": round(total, 1),
        "total_integer": int(scored["value"].sum()),
        "average": round(total / n, 2),
        "max_possible": round(n * (10.9 if p.decimal else 10), 1),
        "tens": int((scored["value"] == 10).sum()),
        "x_count": int((scored["label"] == "X").sum()),
        "misses": int((scored["label"] == "M").sum()),
    }
    if n >= 5:
        cx, cy = float(scored["x"].mean()), float(scored["y"].mean())
        rad = np.hypot(scored["x"] - cx, scored["y"] - cy)
        k = 1000.0 / (p.distance_m * 1000.0)        # mm -> mrad at this distance
        out.update({
            "centre_x_mm": round(cx, 2), "centre_y_mm": round(cy, 2),
            "offset_mm": round(math.hypot(cx, cy), 2),
            "cep50_mm": round(float(np.median(rad)), 2), "cep90_mm": round(float(np.percentile(rad, 90)), 2),
            "cep50_mrad": round(float(np.median(rad)) * k, 3), "cep90_mrad": round(float(np.percentile(rad, 90)) * k, 3),
            "correction": correction_text(p, cx, cy),
        })
    return out


def correction_text(p: TargetProfile, cx: float, cy: float) -> str:
    """Plain description of the group offset and which way to correct (no numbers are invented)."""
    if math.hypot(cx, cy) < 0.25 * p.ring_width_mm:
        return "The group is centred: no sight change needed."
    h = f"{abs(cx):.1f} mm {'right' if cx > 0 else 'left'}" if abs(cx) >= 0.05 else ""
    v = f"{abs(cy):.1f} mm {'high' if cy > 0 else 'low'}" if abs(cy) >= 0.05 else ""
    where = " and ".join(t for t in (h, v) if t)
    if p.sport == "archery":
        tip = "Archers move the sight pin towards the group (\"follow the arrow\")."
    else:
        tip = "Adjust the sights so the impacts move the opposite way by the same amount."
    return f"Group centre is {where} of the target centre. {tip}"


# ------------------------------------------------------------------ cross-sport comparison
def difficulty_table() -> list[dict]:
    """Angular size (milliradians) of each sport's top zone: radius of the effective zone / distance."""
    rows = []
    for p in PROFILES.values():
        eff = (p.x_radius_mm or p.ten_radius_mm) + p.proj_r
        top = "inner 10 / X" if p.x_radius_mm else "10 ring"
        rows.append({"name": f"{p.name} · {top}", "sport": p.sport, "radius_mm": round(eff, 2),
                     "distance_m": p.distance_m, "mrad": round(eff / p.distance_m, 3),
                     "note": "incl. projectile radius"})
        if p.x_radius_mm:
            eff10 = p.ten_radius_mm + p.proj_r
            rows.append({"name": f"{p.name} · 10 ring", "sport": p.sport, "radius_mm": round(eff10, 2),
                         "distance_m": p.distance_m, "mrad": round(eff10 / p.distance_m, 3),
                         "note": "incl. projectile radius"})
    for name, sport, rad, dist, note in REFERENCE_TARGETS:
        rows.append({"name": name, "sport": sport, "radius_mm": rad, "distance_m": dist,
                     "mrad": round(rad / dist, 3), "note": note})
    return sorted(rows, key=lambda r: r["mrad"])


def profile_json(p: TargetProfile) -> dict:
    d = asdict(p)
    d.update({"labels": p.labels, "outer_mm": p.outer_mm, "scoring_limit_mm": p.scoring_limit_mm,
              "rings": [{"value": k, "radius_mm": p.ring_radius(k), "colour": p.colours[k]} for k in range(1, 11)]})
    d["colours"] = {str(k): v for k, v in p.colours.items()}
    return d