"""Builds notebooks/01_dataset_preprocessing_eda.ipynb (run once)."""
from pathlib import Path

import nbformat as nbf

cells = [
    ("md", "# AI DartVision — Dataset, Preprocessing & EDA\n"
           "Coordinates in. Scoring regions out.\n\n"
           "This notebook walks through objectives 01–02 of the proposal: building the labelled dataset "
           "and exploring it before model training."),
    ("code", "import sys; sys.path.append('..')\n"
             "import matplotlib.pyplot as plt\n%matplotlib inline\n"
             "from dartvision.board import score_point, LABELS\n"
             "from dartvision.data import make_synthetic_dataset, make_uniform_dataset, simulate_players\n"
             "from dartvision.preprocessing import validate_coordinates, add_features\n"
             "from dartvision import plots"),
    ("md", "## 1. Geometry scorer (reference baseline)"),
    ("code", "for pt in [(0,0), (0,0.61), (0,0.97), (0.5,0), (0,-0.5), (1.2,0)]:\n    print(pt, score_point(*pt))"),
    ("md", "## 2. Data collection — stratified synthetic coordinates"),
    ("code", "df = make_synthetic_dataset(n_per_class=250, seed=42)\nprint(df.shape, df['label'].nunique(), 'classes')\ndf.head()"),
    ("md", "## 3. Preprocessing — validation and feature engineering"),
    ("code", "df, report = validate_coordinates(df)\nprint(report)\ndf = add_features(df)\ndf[['x','y','r','sin_t','cos_t','label','score']].describe()"),
    ("md", "## 4. EDA — coverage, region mix, spatial patterns"),
    ("code", "plots.landing_map(df.sample(2000, random_state=0), title='Landing coordinates (sample)', s=5); plt.show()"),
    ("code", "plots.region_distribution(df); plt.show()"),
    ("code", "uni = add_features(make_uniform_dataset())\nplots.region_distribution(uni, 'Uniform coverage: natural imbalance'); plt.show()\n"
             "plots.score_distribution(uni); plt.show()"),
    ("md", "Radius alone separates the rings; angle separates the sectors — which is why `r`, `sin θ`, `cos θ` are useful features."),
    ("code", "fig, ax = plt.subplots(figsize=(10,4))\nring = df['label'].str[0].replace({'I':'Bull','O':'Bull','M':'Miss'})\n"
             "for k, d in df.groupby(ring):\n    ax.hist(d['r'], bins=80, alpha=.6, label=k)\nax.legend(); ax.set_xlabel('r (normalised)'); plt.show()"),
    ("md", "## 5. Simulated player throws"),
    ("code", "throws = simulate_players()\nplots.landing_map(throws, hue='player', title='Simulated players', s=8); plt.show()\n"
             "plots.density_heatmap(throws); plt.show()"),
    ("md", "Next: `python run_pipeline.py` trains the five classifiers and writes `results/evaluation_report.md`."),
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
out = Path(__file__).parent / "notebooks" / "01_dataset_preprocessing_eda.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)