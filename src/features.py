"""Feature engineering, following the Stanford reference paper (Khosla et al.).

- numeric features known at launch (goal, duration, text lengths, launch month/weekday)
- out-of-fold target-mean encoding of categorical variables, plus the project's goal
  relative to the average goal of its category (avoids target leakage)
- out-of-fold Naive Bayes (TF-IDF) probability from the project name + blurb
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

CAT_COLS = ["category", "parent_category", "country", "currency"]
BASE_COLS = ["log_goal", "goal_usd", "duration_days", "blurb_len", "blurb_words",
             "name_len", "launch_month", "launch_dow"]


def _prep(df):
    d = df.copy().reset_index(drop=True)
    name, blurb = d["name"].fillna(""), d["blurb"].fillna("")
    d["log_goal"] = np.log1p(d["goal_usd"])
    d["text"] = (name + " " + blurb).str.lower()
    d["blurb_len"] = blurb.str.len()
    d["blurb_words"] = blurb.str.split().str.len()
    d["name_len"] = name.str.len()
    return d


class FeatureBuilder:
    def __init__(self, min_count=30, n_splits=5, seed=42):
        self.min_count, self.n_splits, self.seed = min_count, n_splits, seed

    # ---- categorical target encoding -------------------------------------------------
    def _enc_fit(self, d, y):
        maps = {}
        for c in CAT_COLS:
            g = (pd.DataFrame({"k": d[c].values, "y": y, "g": d["log_goal"].values})
                 .groupby("k").agg(n=("y", "size"), m=("y", "mean"), g=("g", "mean")))
            g = g[g["n"] >= self.min_count]  # rare levels fall back to the global mean
            maps[c] = (g["m"].to_dict(), g["g"].to_dict())
        return maps, float(np.mean(y)), float(d["log_goal"].mean())

    def _enc_apply(self, d, state):
        maps, global_y, global_g = state
        out = pd.DataFrame(index=d.index)
        for c in CAT_COLS:
            y_map, g_map = maps[c]
            out[f"{c}_target_mean"] = d[c].map(y_map).fillna(global_y).astype(float)
            out[f"{c}_goal_diff"] = d["log_goal"] - d[c].map(g_map).fillna(global_g).astype(float)
        return out

    # ---- text: TF-IDF + Multinomial Naive Bayes --------------------------------------
    @staticmethod
    def _nb_fit(text, y):
        vec = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.35)
        nb = MultinomialNB().fit(vec.fit_transform(text), y)
        return vec, nb

    @staticmethod
    def _nb_apply(text, state):
        vec, nb = state
        return nb.predict_proba(vec.transform(text))[:, 1]

    @staticmethod
    def _assemble(d, enc, nb):
        X = pd.concat([d[BASE_COLS], enc], axis=1)
        X["nb_prob"] = nb
        return X

    # ---- public API ------------------------------------------------------------------
    def fit_transform(self, df, y):
        """Training features: encodings / NB probabilities are computed out-of-fold."""
        d, y = _prep(df), np.asarray(y)
        enc_parts, nb = [], np.zeros(len(d))
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(d, y):
            enc_parts.append(self._enc_apply(d.iloc[va], self._enc_fit(d.iloc[tr], y[tr])))
            nb[va] = self._nb_apply(d["text"].iloc[va], self._nb_fit(d["text"].iloc[tr], y[tr]))
        enc = pd.concat(enc_parts).sort_index()
        # states fitted on the full training set are used for test / new data
        self.enc_state_ = self._enc_fit(d, y)
        self.nb_state_ = self._nb_fit(d["text"], y)
        return self._assemble(d, enc, nb)

    def transform(self, df):
        d = _prep(df)
        return self._assemble(d, self._enc_apply(d, self.enc_state_),
                              self._nb_apply(d["text"], self.nb_state_))
