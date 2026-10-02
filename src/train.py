"""Train and evaluate models for Kickstarter success prediction.

Usage:  python src/train.py --data-dir data/raw [--sample 50000]
"""
import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_data
from features import FeatureBuilder


def get_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=30, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=64,
            class_weight="balanced", random_state=42),
    }


def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Precision_1": precision_score(y_true, y_pred),
        "Precision_0": precision_score(y_true, y_pred, pos_label=0),
        "Recall_1": recall_score(y_true, y_pred),
        "Recall_0": recall_score(y_true, y_pred, pos_label=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--sample", type=int, default=None, help="limit number of projects (faster)")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.data_dir, args.sample)
    print(f"Loaded {len(df)} projects | success rate {df['y'].mean():.2f}")

    y = df["y"].to_numpy()
    raw_tr, raw_te, y_tr, y_te = train_test_split(
        df.drop(columns="y"), y, test_size=0.3, stratify=y, random_state=42)

    builder = FeatureBuilder()
    X_tr = builder.fit_transform(raw_tr, y_tr)
    X_te = builder.transform(raw_te)
    print(f"Train {len(X_tr)} / Test {len(X_te)} | {X_tr.shape[1]} features")

    rows, fitted = {}, {}
    for name, model in get_models().items():
        print(f"Training {name} ...")
        model.fit(X_tr, y_tr)
        fitted[name] = model
        rows[name] = evaluate(y_te, model.predict(X_te))

    # Ablation (as in the paper): gradient boosting without the text (Naive Bayes) feature
    cols = [c for c in X_tr.columns if c != "nb_prob"]
    abl = get_models()["Gradient Boosting"].fit(X_tr[cols], y_tr)
    rows["Gradient Boosting (no text feature)"] = evaluate(y_te, abl.predict(X_te[cols]))

    results = pd.DataFrame(rows).T.round(3).sort_values("Accuracy", ascending=False)
    results.to_csv(os.path.join(args.out_dir, "results.csv"))
    print("\n", results.to_string(), "\n")

    best = next(n for n in results.index if n in fitted)
    print("Best model:", best)

    ConfusionMatrixDisplay.from_predictions(
        y_te, fitted[best].predict(X_te), display_labels=["Failed", "Successful"], cmap="Blues")
    plt.title(f"Confusion matrix - {best}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    imp = pd.Series(fitted["Random Forest"].feature_importances_, index=X_tr.columns).sort_values()
    imp.plot.barh(figsize=(7, 6), title="Random Forest feature importance")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "feature_importance.png"), dpi=150)
    plt.close()

    joblib.dump({
        "builder": builder,
        "model": fitted[best],
        "model_name": best,
        "categories": sorted(df["category"].unique()),
        "cat_parent": df.groupby("category")["parent_category"].agg(lambda s: s.mode().iat[0]).to_dict(),
        "countries": sorted(df["country"].unique()),
        "currencies": sorted(df["currency"].unique()),
    }, os.path.join(args.models_dir, "best_model.joblib"))
    print(f"Saved model to {args.models_dir}/best_model.joblib and plots/results to {args.out_dir}/")


if __name__ == "__main__":
    main()
