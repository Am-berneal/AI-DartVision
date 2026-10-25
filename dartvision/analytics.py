"""
Player and session analytics computed only from recorded throws.

Every function takes a table of throws with at least x, y (normalised board units), label and score.
Optional columns unlock more metrics:
    player, session          -> per-player and per-session views
    throw                    -> dart order inside a session (fatigue, streaks, 3-dart groups)
    aim_label, aim_x, aim_y  -> hit rate, bias and error against the target
    visit_id, dart           -> real 3-dart visits (DeepDarts)
A metric that the data cannot support is returned as None, never guessed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .board import LABELS, SECTORS, region_center

MM = 170.0                      # 1.0 normalised unit = 170 mm
MIN_DARTS_GROUPING = 5          # fewer darts than this -> no grouping radius
ZONES = ["treble", "double", "single", "bull", "miss"]
VISIT_BINS = list(range(0, 181, 20))   # 0-19, 20-39, ... 160-180


# ------------------------------------------------------------------ helpers
def _num(v, nd=2):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, nd)


def zone_of(label: str) -> str:
    if label in ("IB", "OB"):
        return "bull"
    if label == "MISS":
        return "miss"
    return {"T": "treble", "D": "double", "S": "single"}.get(str(label)[:1], "single")


def has_aim(df: pd.DataFrame) -> bool:
    return {"aim_label", "aim_x", "aim_y"} <= set(df.columns) and df["aim_label"].notna().any()


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("session", "throw") if c in df.columns]
    return df.sort_values(keys, kind="stable") if keys else df


def zone_mix(df: pd.DataFrame) -> dict:
    n = len(df)
    z = df["label"].map(zone_of) if n else pd.Series(dtype=str)
    counts = {k: int((z == k).sum()) for k in ZONES}
    return {"darts": n, "counts": counts,
            "shares": {k: (_num(counts[k] / n, 4) if n else None) for k in ZONES}}


def sector_counts(df: pd.DataFrame) -> dict:
    """Darts per board number (singles + doubles + trebles), in clockwise board order from 20."""
    labels = df["label"].astype(str)
    num = labels.where(labels.str.match(r"^[SDT]\d+$")).str.slice(1)
    vc = num.dropna().astype(int).value_counts()
    sectors = [{"number": s, "count": int(vc.get(s, 0))} for s in SECTORS]
    return {"sectors": sectors, "bull": int(labels.isin(["IB", "OB"]).sum()),
            "miss": int((labels == "MISS").sum()), "darts": int(len(df))}


def grouping(df: pd.DataFrame) -> dict | None:
    """Group centre and the radii (mm) around it containing 50% / 90% of darts (CEP50 / CEP90)."""
    if len(df) < MIN_DARTS_GROUPING:
        return None
    cx, cy = float(df["x"].mean()), float(df["y"].mean())
    d = np.hypot(df["x"] - cx, df["y"] - cy) * MM
    return {"centre_x": _num(cx, 4), "centre_y": _num(cy, 4),
            "cep50_mm": _num(np.median(d), 1), "cep90_mm": _num(np.percentile(d, 90), 1)}


def _aim_stats(df: pd.DataFrame) -> dict:
    if not has_aim(df) or df.empty:
        return {"hit_rate": None, "bias_x_mm": None, "bias_y_mm": None, "mean_error_mm": None}
    dx, dy = (df["x"] - df["aim_x"]) * MM, (df["y"] - df["aim_y"]) * MM
    return {"hit_rate": _num((df["label"] == df["aim_label"]).mean(), 4),
            "bias_x_mm": _num(dx.mean(), 1), "bias_y_mm": _num(dy.mean(), 1),
            "mean_error_mm": _num(np.hypot(dx, dy).mean(), 1)}


def visit_totals(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """3-dart totals. Real visits when visit_id exists, otherwise consecutive darts grouped in threes
    (per player and session, in throw order). Returns (totals, method)."""
    if df.empty:
        return pd.Series(dtype=float), "none"
    if "visit_id" in df.columns:
        return df.groupby("visit_id")["score"].sum(), "recorded visits"
    t = _ordered(df).copy()
    keys = [c for c in ("player", "session") if c in t.columns]
    pos = t.groupby(keys).cumcount() if keys else pd.Series(np.arange(len(t)), index=t.index)
    t["_grp"] = (pos // 3).values
    sizes = t.groupby(keys + ["_grp"])["score"].transform("size") if keys else t.groupby("_grp")["score"].transform("size")
    t = t[sizes == 3]                          # incomplete trailing groups are not counted
    if t.empty:
        return pd.Series(dtype=float), "consecutive darts grouped in threes"
    return t.groupby(keys + ["_grp"])["score"].sum(), "consecutive darts grouped in threes"


def visit_summary(df: pd.DataFrame) -> dict:
    tot, method = visit_totals(df)
    if tot.empty:
        return {"method": method, "visits": 0, "mean": None, "std": None, "best": None,
                "count_60": 0, "count_100": 0, "count_140": 0, "count_180": 0, "histogram": []}
    bins = VISIT_BINS
    hist = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (tot >= lo) & ((tot < hi) if hi < 180 else (tot <= hi))
        hist.append({"lo": lo, "hi": hi, "count": int(mask.sum())})
    return {"method": method, "visits": int(len(tot)), "mean": _num(tot.mean(), 1), "std": _num(tot.std(), 1),
            "best": int(tot.max()), "count_60": int((tot >= 60).sum()), "count_100": int((tot >= 100).sum()),
            "count_140": int((tot >= 140).sum()), "count_180": int((tot == 180).sum()), "histogram": hist}


def longest_hit_streak(df: pd.DataFrame) -> int | None:
    if not has_aim(df) or "throw" not in df.columns or df.empty:
        return None
    best = 0
    for _, s in _ordered(df).groupby("session" if "session" in df.columns else np.zeros(len(df))):
        run = 0
        for hit in (s["label"] == s["aim_label"]):
            run = run + 1 if hit else 0
            best = max(best, run)
    return int(best)


def session_table(df: pd.DataFrame) -> list[dict]:
    if "session" not in df.columns or df.empty:
        return []
    rows = []
    for sess, s in df.groupby("session", sort=True):
        zm = zone_mix(s)["shares"]
        g = grouping(s)
        a = _aim_stats(s)
        vs = visit_summary(s)
        rows.append({
            "session": sess.item() if hasattr(sess, "item") else sess,
            "darts": int(len(s)),
            "ppd": _num(s["score"].mean(), 2),
            "score_std": _num(s["score"].std(), 1),
            "visit_avg": vs["mean"],
            "cep50_mm": g["cep50_mm"] if g else None,
            "centre_x": g["centre_x"] if g else None, "centre_y": g["centre_y"] if g else None,
            **a,
            **{f"{k}_rate": zm[k] for k in ZONES},
        })
    return rows


def trend_slope(sessions: list[dict], key: str = "ppd") -> float | None:
    pts = [(i, r[key]) for i, r in enumerate(sessions) if r.get(key) is not None]
    if len(pts) < 3:
        return None
    x, y = np.array(pts, dtype=float).T
    return _num(np.polyfit(x, y, 1)[0], 3)


def fatigue(df: pd.DataFrame) -> list[dict] | None:
    """Early / middle / late thirds of each session, by dart order."""
    if "throw" not in df.columns or "session" not in df.columns or df.empty:
        return None
    parts = {0: [], 1: [], 2: []}
    for _, s in _ordered(df).groupby("session"):
        if len(s) < 9:
            continue
        for i, idx in enumerate(np.array_split(np.arange(len(s)), 3)):
            parts[i].append(s.iloc[idx])
    if not parts[0]:
        return None
    out = []
    for i, name in enumerate(["Early", "Middle", "Late"]):
        c = pd.concat(parts[i])
        out.append({"phase": name, "darts": int(len(c)), "ppd": _num(c["score"].mean(), 2),
                    "hit_rate": _aim_stats(c)["hit_rate"]})
    return out


# ------------------------------------------------------------------ players
RADAR = [  # key, label, higher is better
    ("ppd", "Points per dart", True),
    ("hit_rate", "Target hit rate", True),
    ("cep50_mm", "Tight grouping", False),
    ("session_variation", "Steadiness", False),
    ("treble_rate", "Treble rate", True),
    ("improvement", "Improvement", True),
]


def player_core(df: pd.DataFrame) -> dict:
    sessions = session_table(df)
    g = grouping(df)
    ppd_sessions = [r["ppd"] for r in sessions if r["ppd"] is not None]
    return {
        "darts": int(len(df)),
        "sessions": len(sessions),
        "aim": str(df["aim_label"].iloc[0]) if has_aim(df) else None,
        "ppd": _num(df["score"].mean(), 2),
        "three_dart_avg": _num(df["score"].mean() * 3, 1),
        **_aim_stats(df),
        "cep50_mm": g["cep50_mm"] if g else None,
        "cep90_mm": g["cep90_mm"] if g else None,
        "session_variation": _num(np.std(ppd_sessions, ddof=1), 2) if len(ppd_sessions) >= 2 else None,
        "treble_rate": zone_mix(df)["shares"]["treble"],
        "miss_rate": zone_mix(df)["shares"]["miss"],
        "improvement": trend_slope(sessions),
        "longest_hit_streak": longest_hit_streak(df),
    }


def _normalise(values: dict, higher_better: bool) -> dict:
    vals = [v for v in values.values() if v is not None]
    if not vals:
        return {k: None for k in values}
    lo, hi = min(vals), max(vals)
    out = {}
    for k, v in values.items():
        if v is None:
            out[k] = None
        elif hi == lo:
            out[k] = 0.5
        else:
            n = (v - lo) / (hi - lo)
            out[k] = round(n if higher_better else 1 - n, 3)
    return out


def roster_profile(throws: pd.DataFrame) -> dict:
    """Core metrics for every player, plus each metric scaled 0-1 within the roster (1 = best)."""
    cores = {p: player_core(d) for p, d in throws.groupby("player")}
    scaled = {p: {} for p in cores}
    for key, _, hb in RADAR:
        norm = _normalise({p: c[key] for p, c in cores.items()}, hb)
        for p in cores:
            scaled[p][key] = norm[p]
    return {"cores": cores, "scaled": scaled}


def player_analytics(throws: pd.DataFrame, player: str, profile: dict | None = None) -> dict:
    """profile = roster_profile(throws), passed in when the caller has it cached."""
    df = throws[throws["player"] == player]
    if df.empty:
        raise KeyError(player)
    prof = profile or roster_profile(throws)
    core = prof["cores"][player]
    avg_scaled = {}
    for key, _, _ in RADAR:
        vals = [prof["scaled"][p][key] for p in prof["scaled"] if prof["scaled"][p][key] is not None]
        avg_scaled[key] = _num(np.mean(vals), 3) if vals else None
    ranks = {}
    for key, _, hb in RADAR:
        vals = {p: c[key] for p, c in prof["cores"].items() if c[key] is not None}
        if core[key] is None or len(vals) < 2:
            ranks[key] = None
            continue
        order = sorted(vals, key=lambda p: vals[p], reverse=hb)
        ranks[key] = {"rank": order.index(player) + 1, "of": len(order)}
    return {
        "player": player,
        "core": core,
        "zones": zone_mix(df),
        "sectors": sector_counts(df),
        "grouping": grouping(df),
        "sessions": session_table(df),
        "visits": visit_summary(df),
        "fatigue": fatigue(df),
        "radar": {"axes": [{"key": k, "label": l, "higher_is_better": hb} for k, l, hb in RADAR],
                  "player": prof["scaled"][player], "roster_avg": avg_scaled, "ranks": ranks,
                  "roster_size": len(prof["cores"])},
        "aim": {"x": _num(df["aim_x"].iloc[0], 4), "y": _num(df["aim_y"].iloc[0], 4)} if has_aim(df) else None,
    }


COMPARE = [  # key, label, unit, higher is better
    ("ppd", "Points per dart", "pts", True),
    ("three_dart_avg", "3-dart average", "pts", True),
    ("hit_rate", "Target hit rate", "%", True),
    ("mean_error_mm", "Mean error from target", "mm", False),
    ("cep50_mm", "Grouping radius (CEP50)", "mm", False),
    ("cep90_mm", "Grouping radius (CEP90)", "mm", False),
    ("treble_rate", "Treble rate", "%", True),
    ("miss_rate", "Miss rate", "%", False),
    ("session_variation", "Session-to-session variation", "pts/dart", False),
    ("improvement", "Improvement per session", "pts/dart", True),
    ("longest_hit_streak", "Longest target-hit streak", "darts", True),
]


def compare_players(throws: pd.DataFrame, a: str, b: str) -> dict:
    da, db = throws[throws["player"] == a], throws[throws["player"] == b]
    if da.empty:
        raise KeyError(a)
    if db.empty:
        raise KeyError(b)
    ca, cb = player_core(da), player_core(db)
    rows = []
    for key, label, unit, hb in COMPARE:
        va, vb = ca[key], cb[key]
        if va is None or vb is None:
            better = None
        elif va == vb:
            better = "tie"
        else:
            better = "a" if (va > vb) == hb else "b"
        rows.append({"key": key, "label": label, "unit": unit, "higher_is_better": hb,
                     "a": va, "b": vb, "better": better})
    sa, sb = session_table(da), session_table(db)
    return {"a": a, "b": b, "aim_a": ca["aim"], "aim_b": cb["aim"], "same_aim": ca["aim"] == cb["aim"],
            "metrics": rows,
            "sessions": {"a": [{"session": r["session"], "ppd": r["ppd"], "hit_rate": r["hit_rate"]} for r in sa],
                         "b": [{"session": r["session"], "ppd": r["ppd"], "hit_rate": r["hit_rate"]} for r in sb]}}


# ------------------------------------------------------------------ real data (DeepDarts)
def real_analytics(real: pd.DataFrame, subset: str | None = None) -> dict:
    df = real if not subset or subset == "all" else real[real["subset"] == subset]
    if df.empty:
        return {"subset": subset or "all", "darts": 0, "visits": visit_summary(df), "sectors": sector_counts(df),
                "zones": zone_mix(df), "sessions": [], "dart_order": [], "subsets": []}
    sessions = []
    for sess, s in df.groupby("session"):
        vs = visit_summary(s)
        zm = zone_mix(s)["shares"]
        date = str(s["date"].iloc[0])[:10] if "date" in s.columns else None
        sessions.append({"session": sess, "date": date, "subset": str(s["subset"].iloc[0]),
                         "darts": int(len(s)), "visits": vs["visits"], "ppd": _num(s["score"].mean(), 2),
                         "visit_avg": vs["mean"], "miss_rate": zm["miss"], "treble_rate": zm["treble"],
                         "double_rate": zm["double"]})
    sessions.sort(key=lambda r: (r["date"] or "", r["session"]))
    order = []
    if "dart" in df.columns:
        for d, s in df.groupby("dart"):
            zm = zone_mix(s)["shares"]
            order.append({"dart": int(d), "darts": int(len(s)), "ppd": _num(s["score"].mean(), 2),
                          "miss_rate": zm["miss"], "treble_rate": zm["treble"]})
    subsets = []
    if "subset" in real.columns:
        for name, s in real.groupby("subset"):
            vs = visit_summary(s)
            subsets.append({"subset": name, "darts": int(len(s)), "sessions": int(s["session"].nunique()),
                            "ppd": _num(s["score"].mean(), 2), "visit_avg": vs["mean"],
                            **{f"{k}_rate": v for k, v in zone_mix(s)["shares"].items()}})
    return {"subset": subset or "all", "darts": int(len(df)), "visits": visit_summary(df),
            "sectors": sector_counts(df), "zones": zone_mix(df), "sessions": sessions,
            "session_ppd_slope": trend_slope(sessions), "dart_order": order, "subsets": subsets}


# ------------------------------------------------------------------ batch uploads
def batch_summary(df: pd.DataFrame) -> dict:
    """Per-player summary of uploaded throws. df: x, y (normalised), label, score, + optional player,
    target, session (in upload order)."""
    df = df.copy()
    notes = []
    if "player" not in df.columns or df["player"].fillna("").eq("").all():
        df["player"] = "All throws"
        notes.append("Add a 'player' column to split the summary by player.")
    df["player"] = df["player"].fillna("").replace("", "Unnamed")
    df["throw"] = df.groupby("player").cumcount() + 1
    has_target = "target" in df.columns and df["target"].fillna("").ne("").any()
    if has_target:
        tgt = df["target"].fillna("").astype(str).str.strip().str.upper()
        valid = tgt.isin(LABELS)
        if (~valid & tgt.ne("")).any():
            bad = sorted(set(tgt[~valid & tgt.ne("")]))[:5]
            notes.append(f"Unrecognised target values ignored: {', '.join(bad)} (use labels like T20, D16, IB).")
        df["aim_label"] = tgt.where(valid)
        centres = {lbl: region_center(lbl) for lbl in df["aim_label"].dropna().unique()}
        df["aim_x"] = df["aim_label"].map(lambda l: centres[l][0] if l in centres else np.nan)
        df["aim_y"] = df["aim_label"].map(lambda l: centres[l][1] if l in centres else np.nan)
    else:
        notes.append("Add a 'target' column (e.g. T20) to unlock hit rate, bias and error.")
    has_session = "session" in df.columns and df["session"].fillna("").astype(str).ne("").any()
    if not has_session:
        df = df.drop(columns=["session"], errors="ignore")
        df["session"] = 1
        notes.append("Add a 'session' column to see trends across sessions.")
    else:
        df["session"] = df["session"].fillna("").astype(str)
    players = []
    for p, d in df.groupby("player", sort=False):
        aimed = d.dropna(subset=["aim_label"]) if has_target else d.iloc[0:0]
        a = _aim_stats(aimed) if has_target and not aimed.empty else _aim_stats(d.iloc[0:0])
        g = grouping(d)
        vs = visit_summary(d)
        sessions = session_table(d) if has_session else []
        players.append({
            "player": p, "darts": int(len(d)), "ppd": _num(d["score"].mean(), 2),
            "group_avg": vs["mean"], "groups": vs["visits"],
            "zones": zone_mix(d)["shares"],
            "cep50_mm": g["cep50_mm"] if g else None,
            "targets": sorted(aimed["aim_label"].unique().tolist()) if has_target and not aimed.empty else [],
            **a,
            "sessions": [{"session": r["session"], "darts": r["darts"], "ppd": r["ppd"]} for r in sessions],
        })
    return {"players": players, "notes": notes, "has_target": bool(has_target), "has_session": bool(has_session),
            "grouping_note": "3-dart groups = consecutive darts in upload order, grouped in threes per player and session."}