import numpy as np
import pandas as pd

from dartvision import analytics as A
from dartvision.board import label_to_score, region_center, score_region


def _throws(points, player="P", session=1, aim="T20"):
    ax, ay = region_center(aim)
    df = pd.DataFrame(points, columns=["x", "y"])
    df["player"], df["session"], df["throw"] = player, session, np.arange(1, len(df) + 1)
    df["aim_label"], df["aim_x"], df["aim_y"] = aim, ax, ay
    df["label"] = score_region(df["x"].values, df["y"].values)
    df["score"] = [label_to_score(l) for l in df["label"]]
    return df


T20 = region_center("T20")
S20 = (0.0, 0.8)
MISS = (0.0, 1.1)


def test_zone_mix_counts_and_shares():
    df = _throws([T20, T20, S20, MISS])
    z = A.zone_mix(df)
    assert z["counts"] == {"treble": 2, "double": 0, "single": 1, "bull": 0, "miss": 1}
    assert z["shares"]["treble"] == 0.5


def test_sector_counts_in_board_order():
    df = _throws([T20, S20, region_center("T1"), region_center("IB"), MISS])
    s = A.sector_counts(df)
    assert s["sectors"][0] == {"number": 20, "count": 2}
    assert s["sectors"][1] == {"number": 1, "count": 1}
    assert s["bull"] == 1 and s["miss"] == 1


def test_grouping_needs_five_darts_and_is_zero_for_identical_darts():
    assert A.grouping(_throws([T20] * 4)) is None
    g = A.grouping(_throws([T20] * 6))
    assert g["cep50_mm"] == 0.0 and g["cep90_mm"] == 0.0


def test_visit_groups_of_three_and_milestones():
    df = _throws([T20] * 6 + [S20])           # two full groups of 180, one dart left over
    v = A.visit_summary(df)
    assert v["visits"] == 2 and v["best"] == 180 and v["count_180"] == 2
    assert v["method"] == "consecutive darts grouped in threes"


def test_real_visits_use_visit_id():
    df = _throws([T20, S20, MISS, T20])
    df["visit_id"] = ["v1", "v1", "v1", "v2"]
    v = A.visit_summary(df)
    assert v["method"] == "recorded visits" and v["visits"] == 2 and v["best"] == 80


def test_hit_streak_and_hit_rate():
    df = _throws([T20, T20, S20, T20, T20, T20, S20])
    assert A.longest_hit_streak(df) == 3
    assert A._aim_stats(df)["hit_rate"] == round(5 / 7, 4)


def test_trend_slope_needs_three_sessions():
    assert A.trend_slope([{"ppd": 10}, {"ppd": 20}]) is None
    assert A.trend_slope([{"ppd": 10}, {"ppd": 20}, {"ppd": 30}]) == 10.0


def test_compare_marks_the_better_player():
    a = _throws([T20] * 9, player="A")
    b = _throws([S20] * 9, player="B")
    c = A.compare_players(pd.concat([a, b]), "A", "B")
    ppd = next(m for m in c["metrics"] if m["key"] == "ppd")
    assert ppd["a"] == 60 and ppd["b"] == 20 and ppd["better"] == "a"


def test_batch_summary_handles_missing_columns():
    df = _throws([T20, S20, MISS])[["x", "y", "label", "score"]]
    out = A.batch_summary(df)
    assert out["players"][0]["player"] == "All throws"
    assert out["players"][0]["hit_rate"] is None
    assert any("target" in n for n in out["notes"])


def test_batch_summary_with_target_column():
    df = _throws([T20, T20, S20])[["x", "y", "label", "score"]]
    df["player"], df["target"] = "Me", ["T20", "t20", "T20"]
    p = A.batch_summary(df)["players"][0]
    assert p["hit_rate"] == round(2 / 3, 4) and p["targets"] == ["T20"]