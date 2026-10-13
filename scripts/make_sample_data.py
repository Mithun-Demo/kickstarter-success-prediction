"""Generate a small SYNTHETIC dataset in Web Robots format to check the pipeline runs.

Not real data - never report results from it. Usage:
    python scripts/make_sample_data.py --out data/sample
    python src/train.py --data-dir data/sample
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/sample")
ap.add_argument("--n", type=int, default=6000)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

rng = np.random.default_rng(0)
cats = {"Documentary": "Film & Video", "Tabletop Games": "Games", "Plays": "Theater",
        "Mixed Media": "Art", "Apps": "Technology", "Food Trucks": "Food",
        "Rock": "Music", "Fiction": "Publishing"}
base = dict(zip(cats, [0.7, 0.75, 0.65, 0.55, 0.35, 0.3, 0.55, 0.4]))
good = "new amazing community handmade story original album".split()
bad = "revolutionary ultimate disruptive platform blockchain startup app".split()
countries = {"US": ("USD", 1.0), "GB": ("GBP", 1.27), "CA": ("CAD", 0.75)}

rows = []
for i in range(args.n):
    cat = rng.choice(list(cats))
    goal = float(np.exp(rng.normal(8.5, 1.4)))
    p = np.clip(base[cat] - 0.08 * (np.log(goal) - 8.5), 0.05, 0.95)
    ok = rng.random() < p
    pool = good if (ok and rng.random() < 0.6) else bad
    words = list(rng.choice(pool + ["project", "help", "make", "world"], size=int(rng.integers(8, 20))))
    country = str(rng.choice(list(countries)))
    cur, rate = countries[country]
    launched = int(1.55e9 + rng.integers(0, int(6e7)))
    rows.append({
        "name": f"Project {i} {cat}", "blurb": " ".join(words), "goal": goal,
        "state": "successful" if ok else "failed", "country": country, "currency": cur,
        "launched_at": launched, "deadline": launched + int(rng.integers(15, 60)) * 86400,
        "static_usd_rate": rate,
        "category": json.dumps({"name": cat, "parent_name": cats[cat]}),
    })
pd.DataFrame(rows).to_csv(os.path.join(args.out, "sample.csv"), index=False)
print(f"Wrote {args.n} synthetic rows to {args.out}/sample.csv")
