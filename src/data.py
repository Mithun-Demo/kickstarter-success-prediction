"""Load and standardise Kickstarter data.

Supports the Web Robots scrape (used in the Stanford reference paper) and,
as a fallback, the Kaggle "ks-projects-201801.csv" file (no blurb text there).
"""
import glob
import json
import os

import pandas as pd

LABELS = {"successful": 1, "failed": 0}
NEEDED = {
    "name", "blurb", "goal", "state", "country", "currency", "deadline",
    "launched_at", "static_usd_rate", "category",           # Web Robots
    "main_category", "launched", "usd_goal_real",           # Kaggle
}


def _json_get(s, key):
    try:
        return json.loads(s).get(key)
    except Exception:
        return None


def _from_webrobots(raw):
    df = raw[raw["state"].isin(LABELS)].copy()
    # Files overlap month to month, so drop duplicate projects (as in the paper)
    df = df.drop_duplicates(subset=["name", "blurb", "launched_at", "deadline"])
    launched = pd.to_datetime(df["launched_at"], unit="s")
    cat = df["category"].map(lambda s: _json_get(s, "name"))
    parent = df["category"].map(lambda s: _json_get(s, "parent_name")).fillna(cat)
    return pd.DataFrame({
        "name": df["name"].fillna(""),
        "blurb": df["blurb"].fillna(""),
        "goal_usd": df["goal"] * df["static_usd_rate"].fillna(1.0),
        "duration_days": (df["deadline"] - df["launched_at"]) / 86400,
        "launch_month": launched.dt.month,
        "launch_dow": launched.dt.dayofweek,
        "category": cat,
        "parent_category": parent,
        "country": df["country"],
        "currency": df["currency"],
        "y": df["state"].map(LABELS),
    })


def _from_kaggle(raw):
    df = raw[raw["state"].isin(LABELS)].copy()
    launched = pd.to_datetime(df["launched"])
    deadline = pd.to_datetime(df["deadline"])
    return pd.DataFrame({
        "name": df["name"].fillna(""),
        "blurb": "",
        "goal_usd": df["usd_goal_real"],
        "duration_days": (deadline - launched).dt.total_seconds() / 86400,
        "launch_month": launched.dt.month,
        "launch_dow": launched.dt.dayofweek,
        "category": df["category"],
        "parent_category": df["main_category"],
        "country": df["country"],
        "currency": df["currency"],
        "y": df["state"].map(LABELS),
    })


def load_data(data_dir="data/raw", sample=None, seed=42):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. See the README for how to get the dataset."
        )
    raw = pd.concat(
        [pd.read_csv(f, usecols=lambda c: c in NEEDED, low_memory=False) for f in files],
        ignore_index=True,
    )
    if "static_usd_rate" in raw.columns:
        df = _from_webrobots(raw)
    elif "usd_goal_real" in raw.columns:
        df = _from_kaggle(raw)
    else:
        raise ValueError("Unrecognised CSV format (expected Web Robots or Kaggle columns).")

    df = df.dropna(subset=["goal_usd", "duration_days"])
    df = df[(df["goal_usd"] > 0) & (df["duration_days"] > 0) & (df["duration_days"] <= 100)]
    for c in ["category", "parent_category", "country", "currency"]:
        df[c] = df[c].fillna("unknown").astype(str)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=seed)
    return df.reset_index(drop=True)
