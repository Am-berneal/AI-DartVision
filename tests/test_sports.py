import numpy as np

from dartvision import sports as S


def _score(key, d):
    p = S.PROFILES[key]
    return S.score_shots(p, d, [0.0] * len(d))


def test_air_rifle_decimal_and_pellet_edge():
    s = _score("air_rifle", [0.0, 2.5, 2.51, 25.0, 25.01])
    assert list(s["decimal"]) == [10.9, 10.0, 9.9, 1.0, 0.0]      # pellet edge touching the 10 ring = 10.0
    assert list(s["label"]) == ["10", "10", "9", "1", "M"]


def test_air_pistol_inner_ten_and_ring_edges():
    s = _score("air_pistol", [0.0, 4.75, 4.76, 8.0, 8.01])
    assert list(s["label"]) == ["X", "X", "10", "10", "9"]
    assert s["decimal"].iloc[3] == 10.0 and s["decimal"].iloc[4] == 9.9


def test_archery_line_cutter_scores_higher():
    r = S.ARROW_MM / 2
    s = _score("archery_122", [61 + r, 61 + r + 0.01, 610 + r, 610 + r + 0.01])
    assert list(s["value"]) == [10, 9, 1, 0]


def test_ring_one_is_the_published_outer_diameter():
    assert S.PROFILES["air_rifle"].outer_mm * 2 == 45.5
    assert S.PROFILES["air_pistol"].outer_mm * 2 == 155.5
    assert S.PROFILES["archery_122"].outer_mm * 2 == 1220
    assert S.PROFILES["archery_40"].outer_mm * 2 == 400


def test_dataset_is_balanced_and_labels_match_rules():
    for p in S.PROFILES.values():
        df = S.make_dataset(p, n_per_class=40, seed=1)
        assert set(df["label"].value_counts()) == {40}
        assert (S.score_shots(p, df["x"], df["y"])["label"].values == df["label"].values).all()


def test_group_stats_centred_group():
    p = S.PROFILES["air_rifle"]
    s = S.score_shots(p, [1, -1, 0, 0, 0], [0, 0, 1, -1, 0])
    g = S.group_stats(p, s)
    assert g["offset_mm"] == 0 and g["shots"] == 5 and "centred" in g["correction"]
    assert np.isclose(g["cep50_mrad"], g["cep50_mm"] / 10)