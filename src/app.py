"""Streamlit demo: predict the success probability of a Kickstarter project.

Run:  streamlit run src/app.py   (after running src/train.py)
"""
import os

import joblib
import pandas as pd
import streamlit as st

import features  # noqa: F401  (needed so joblib can unpickle FeatureBuilder)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")

st.set_page_config(page_title="Kickstarter Success Predictor", page_icon="🚀")
st.title("🚀 Kickstarter Success Predictor")
st.caption("Predicts whether a project will reach its funding goal, using only information known at launch.")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

b = load_bundle()

name = st.text_input("Project name", "Pocket Board Game Night")
blurb = st.text_area("Short description (blurb)", "A compact card game for friends that fits in your pocket.")
c1, c2 = st.columns(2)
goal = c1.number_input("Funding goal (USD)", min_value=1.0, value=5000.0, step=500.0)
duration = c2.slider("Campaign duration (days)", 1, 90, 30)
category = st.selectbox("Category", b["categories"])
c3, c4 = st.columns(2)
country = c3.selectbox("Country", b["countries"], index=b["countries"].index("US") if "US" in b["countries"] else 0)
currency = c4.selectbox("Currency", b["currencies"], index=b["currencies"].index("USD") if "USD" in b["currencies"] else 0)
launch = st.date_input("Planned launch date")

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "name": name, "blurb": blurb, "goal_usd": goal, "duration_days": duration,
        "launch_month": launch.month, "launch_dow": launch.weekday(),
        "category": category, "parent_category": b["cat_parent"].get(category, "unknown"),
        "country": country, "currency": currency,
    }])
    prob = float(b["model"].predict_proba(b["builder"].transform(row))[0, 1])
    st.metric("Probability of success", f"{prob:.0%}")
    st.progress(prob)
    st.success("Likely to succeed ✅") if prob >= 0.5 else st.warning("Likely to fall short ⚠️")
    st.caption(f"Model: {b['model_name']}")
