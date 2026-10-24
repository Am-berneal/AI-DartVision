"""
Convert the DeepDarts labels (IEEE DataPort) into real throws in board coordinates.

Usage:
    python prepare_real_data.py                      # reads data/deepdarts/labels.pkl
    python prepare_real_data.py path/to/labels.pkl

Writes data/real_throws.csv (one row per unique dart). Then run: python run_pipeline.py
"""
import sys
from pathlib import Path

from dartvision.deepdarts import calibration_check, load_labels, to_board

ROOT = Path(__file__).parent


def main(path: Path):
    if not path.exists():
        sys.exit(f"Labels file not found: {path}\nDownload the DeepDarts dataset from IEEE DataPort "
                 f"(DOI 10.21227/05e7-xs69) and put labels.pkl in data/deepdarts/")
    labels = load_labels(path)
    print(f"Loaded {len(labels)} labelled photos from {path}")
    print(f"Calibration check (should be ~0): {calibration_check(labels):.2e}")
    real = to_board(labels)
    out = ROOT / "data" / "real_throws.csv"
    real.to_csv(out, index=False)
    print(f"Saved {len(real)} unique darts, {real['visit_id'].nunique()} visits, "
          f"{real['session'].nunique()} sessions -> {out}")
    print(real.groupby("subset").agg(darts=("x", "size"), avg_score=("score", "mean"),
                                     miss_rate=("label", lambda s: (s == "MISS").mean())).round(3))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "deepdarts" / "labels.pkl")