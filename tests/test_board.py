import numpy as np
import pandas as pd

from dartvision.board import LABELS, label_to_score, region_center, score_point, score_region
from dartvision.data import make_synthetic_dataset
from dartvision.preprocessing import normalise, validate_coordinates


def test_bulls_and_miss():
    assert score_point(0, 0) == ("IB", 50)
    assert score_point(0, 0.07) == ("OB", 25)
    assert score_point(1.2, 0) == ("MISS", 0)


def test_known_regions():
    assert score_point(0, 0.61)[0] == "T20"      # 103 mm straight up
    assert score_point(0, 0.97)[0] == "D20"      # 165 mm straight up
    assert score_point(0, -0.5)[0] == "S3"       # straight down
    assert score_point(0.5, 0)[0] == "S6"        # straight right
    assert score_point(-0.5, 0)[0] == "S11"      # straight left


def test_scores():
    assert label_to_score("T20") == 60
    assert label_to_score("D16") == 32
    assert label_to_score("S5") == 5


def test_region_centres_round_trip():
    for lbl in LABELS:
        x, y = region_center(lbl)
        assert score_point(x, y)[0] == lbl, lbl


def test_dataset_covers_all_regions():
    df = make_synthetic_dataset(n_per_class=20, seed=0)
    assert set(df["label"]) == set(LABELS)
    assert (df.groupby("label").size() == 20).all()


def test_normalise_mm_and_validate():
    x, y = normalise([0], [170], unit="mm")
    assert np.isclose(y[0], 1.0)
    bad = pd.DataFrame({"x": [0.1, "a", 5.0], "y": [0.1, 0.2, 5.0]})
    clean, rep = validate_coordinates(bad)
    assert len(clean) == 1 and rep["dropped_non_numeric"] == 1 and rep["dropped_out_of_range"] == 1


def test_vectorised():
    out = score_region(np.array([0, 0, 0.5]), np.array([0, 0.61, 0]))
    assert list(out) == ["IB", "T20", "S6"]