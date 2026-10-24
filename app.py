"""
AI DartVision - Streamlit demonstration app.

Run:  streamlit run app.py
(Run `python run_pipeline.py` first so the trained models and results exist.)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dartvision import plots
from dartvision.board import LABEL_NAMES, label_to_score, score_point
from dartvision.data import simulate_players
from dartvision.players import bias_description, player_summary, session_trend
from dartvision.preprocessing import add_features, feature_matrix, normalise

ROOT = Path(__file__).parent
MODELS, RESULTS, DATA = ROOT / "models", ROOT / "results", ROOT / "data"
FAVICON = ROOT / "frontend" / "favicon.png"
LOGO_FULL = ROOT / "frontend" / "logo-full.png"
LOGO_ICON = ROOT / "frontend" / "logo-icon.png"

st.set_page_config(page_title="AI DartVision",
                   page_icon=str(FAVICON) if FAVICON.exists() else "🎯",
                   layout="wide")
if LOGO_FULL.exists() and LOGO_ICON.exists():
    st.logo(str(LOGO_FULL), icon_image=str(LOGO_ICON))


@st.cache_resource
def load_models():
    found = {}
    for f in sorted(MODELS.glob("*_*.joblib")):
        if f.name == "geometry_baseline.joblib":
            continue
        obj = joblib.load(f)
        found[f"{obj['name']} ({obj['feature_set']})"] = obj
    return found


@st.cache_data
def load_throws():
    p = DATA / "player_throws.csv"
    return pd.read_csv(p) if p.exists() else simulate_players()


models = load_models()
st.title("🎯 AI DartVision")
st.caption("Coordinates in. Scoring regions out. — ML-based dartboard score prediction & player performance analysis")

if not models:
    st.error("No trained models found. Open a terminal in the project folder and run:  `python run_pipeline.py`")
    st.stop()

best_key = None
bm = MODELS / "best_model.json"
if bm.exists():
    b = json.loads(bm.read_text())
    best_key = f"{b['best_model']} ({b['best_feature_set']})"

with st.sidebar:
    st.header("Settings")
    keys = list(models)
    model_key = st.selectbox("Classifier", keys, index=keys.index(best_key) if best_key in keys else 0)
    unit = st.radio("Coordinate unit", ["Normalised (board edge = 1.0)", "Millimetres from centre"])
    st.markdown("---")
    st.caption("Board centre = (0, 0). x → right, y → up (20 is at the top). "
               "All data is synthetic; the geometry scorer is the reference.")

tab_pred, tab_player, tab_models, tab_batch = st.tabs(
    ["Predict a throw", "Player analysis", "Model scorecard", "Batch scoring (CSV)"])

# --------------------------------------------------------------- PREDICT
with tab_pred:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("Landing coordinates")
        is_mm = unit.startswith("Milli")
        lim, step = (200.0, 1.0) if is_mm else (1.2, 0.01)
        if "xy" not in st.session_state:
            st.session_state.xy = (0.0, 0.61) if not is_mm else (0.0, 103.0)
        if st.button("🎲 Random throw"):
            rng = np.random.default_rng()
            r = 1.12 * np.sqrt(rng.uniform()); t = rng.uniform(0, 2 * np.pi)
            v = (r * np.cos(t), r * np.sin(t))
            st.session_state.xy = (v[0] * 170, v[1] * 170) if is_mm else v
        x_in = st.number_input("X", -lim, lim, float(np.clip(st.session_state.xy[0], -lim, lim)), step)
        y_in = st.number_input("Y", -lim, lim, float(np.clip(st.session_state.xy[1], -lim, lim)), step)
        x, y = normalise([x_in], [y_in], unit="mm" if is_mm else "norm")
        x, y = float(x[0]), float(y[0])

        m = models[model_key]
        X = feature_matrix(add_features(pd.DataFrame({"x": [x], "y": [y]})), m["feature_set"])
        pred = str(m["model"].predict(X)[0])
        geo_label, geo_score = score_point(x, y)

        st.markdown("---")
        a, b, c = st.columns(3)
        a.metric("Predicted region", pred, LABEL_NAMES.get(pred, ""), delta_color="off")
        b.metric("Score", label_to_score(pred))
        c.metric("Geometry rule", f"{geo_label} = {geo_score}")
        if pred == geo_label:
            st.success(f"{model_key} agrees with the geometry baseline.")
        else:
            st.warning(f"{model_key} disagrees with the geometry baseline (the rule says {geo_label}). "
                       "This usually happens close to a wire.")
        st.caption(f"Normalised: x={x:.3f}, y={y:.3f}, r={np.hypot(x, y):.3f}, "
                   f"angle={np.degrees(np.arctan2(y, x)):.1f}°")
    with c2:
        fig, ax = plt.subplots(figsize=(6, 6))
        plots.draw_board(ax)
        ax.scatter([x], [y], s=160, marker="X", color="#ffcc00", edgecolor="black", linewidth=1.2, zorder=10)
        st.pyplot(fig, clear_figure=True)

# --------------------------------------------------------------- PLAYERS
with tab_player:
    st.subheader("Player-level accuracy (simulated players)")
    st.info("These players are simulated for demonstration, not recorded from real people. "
            "The 'Batch scoring' tab can score your own recorded throws, but meaningful player "
            "trends need real sessions recorded over time.")
    throws = load_throws()
    summary = player_summary(throws)
    summary["bias"] = [bias_description(bx, by) for bx, by in zip(summary["bias_x_mm"], summary["bias_y_mm"])]
    player = st.selectbox("Player", list(summary["player"]))
    row = summary[summary["player"] == player].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Hit rate on {row['aim']}", f"{row['hit_rate_target']:.0%}")
    k2.metric("3-dart average", f"{row['three_dart_avg']:.1f}")
    k3.metric("Mean error", f"{row['mean_error_mm']:.1f} mm")
    k4.metric("Tendency", row["bias"])
    pt = throws[throws["player"] == player]
    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        plots.landing_map(pt, hue="session", title=f"{player} - landings by session", ax=ax, s=12)
        st.pyplot(fig, clear_figure=True)
    with g2:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        plots.density_heatmap(pt, bins=50, title=f"{player} - density", ax=ax)
        st.pyplot(fig, clear_figure=True)
    trend = session_trend(throws)
    st.markdown("**Session trend**")
    st.line_chart(trend[trend["player"] == player].set_index("session")[["avg_score", "hit_rate"]])
    st.markdown("**All players**")
    st.dataframe(summary, width="stretch", hide_index=True)

# --------------------------------------------------------------- MODELS
with tab_models:
    st.subheader("Model scorecard")
    mp = RESULTS / "metrics.csv"
    if mp.exists():
        met = pd.read_csv(mp)
        which = st.radio("Evaluation set", ["test", "uniform"], horizontal=True,
                         format_func=lambda s: {"test": "Held-out stratified test", "uniform": "Uniform board coverage"}[s])
        show = met[met["eval_set"] == which][["model", "feature_set", "accuracy", "macro_f1", "macro_precision",
                                              "macro_recall", "score_mae", "boundary_accuracy", "train_seconds"]]
        st.dataframe(show.sort_values("macro_f1", ascending=False).round(4), width="stretch", hide_index=True)
        img = RESULTS / "figures" / "model_comparison.png"
        if img.exists():
            st.image(str(img))
        cm_files = sorted((RESULTS / "figures").glob("confusion_*.png"))
        if cm_files:
            pick = st.selectbox("Confusion matrix", cm_files, format_func=lambda p: p.stem.replace("confusion_", ""))
            st.image(str(pick))
        st.caption("The geometry baseline is 100% by construction: it is the rule set that produced the labels. "
                   "ML is assessed for agreement with it and robustness near boundaries.")
    else:
        st.warning("Run `python run_pipeline.py` to generate metrics.")

# --------------------------------------------------------------- BATCH
with tab_batch:
    st.subheader("Score a CSV of throws")
    st.write("Upload a CSV with columns `x`, `y` (and optionally `player`). Units follow the sidebar setting.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if not {"x", "y"}.issubset(df.columns):
            st.error("CSV needs columns x and y.")
        else:
            nx, ny = normalise(df["x"], df["y"], unit="mm" if unit.startswith("Milli") else "norm")
            work = add_features(pd.DataFrame({"x": nx, "y": ny}))
            m = models[model_key]
            df["predicted_region"] = m["model"].predict(feature_matrix(work, m["feature_set"]))
            df["predicted_score"] = [label_to_score(l) for l in df["predicted_region"]]
            df["geometry_region"] = [score_point(a, b)[0] for a, b in zip(nx, ny)]
            df["agrees"] = df["predicted_region"] == df["geometry_region"]
            st.dataframe(df, width="stretch")
            st.write(f"Total score: **{df['predicted_score'].sum()}**  ·  "
                     f"Agreement with geometry: **{df['agrees'].mean():.1%}**")
            fig, ax = plt.subplots(figsize=(6, 6))
            plots.landing_map(pd.DataFrame({"x": nx, "y": ny}), title="Uploaded throws", ax=ax, s=18)
            st.pyplot(fig, clear_figure=True)
            st.download_button("Download scored CSV", df.to_csv(index=False), "scored_throws.csv", "text/csv")