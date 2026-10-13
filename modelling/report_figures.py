"""Figures and summary tables for the Chapter 4 report (run after land_price_models.py and compare_models_cv.py)."""
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
src = (HERE / "land_price_models.py").read_text(encoding="utf-8").split("cv = GroupKFold(5)")[0]
src = src.replace('log = open(OUT / "run_log.txt", "w", encoding="utf-8")', "log = io.StringIO()")
ns = {"__file__": str(HERE / "land_price_models.py"), "io": io, "print": lambda *a, **k: None}
exec(src, ns)
X, y, d = ns["X"], ns["y"], ns["d"]
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# Descriptive statistics of the modelling dataset
v = np.exp(y)
desc = {"n": len(X), "median_value": float(np.median(v)), "mean_value": float(v.mean()),
        "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max()),
        "median_ha": float(np.median(d.ha)), "share_2023": float(X.year_2023.mean()),
        "dist_km_n": int(X.dist_km.notna().sum()), "road_n": int(X.road_access.sum()),
        "amenity_n": int(X.near_amenity.sum()), "desc_n": int(X.has_description.sum()),
        "skew_raw": float(pd.Series(v).skew()), "skew_log": float(pd.Series(y).skew())}
loc = d.groupby("location").agg(n=("value", "size"), median_value=("value", "median"), median_ha=("ha", "median")).sort_values("n", ascending=False)
loc.to_csv(OUT / "location_summary.csv")
json.dump(desc, open(OUT / "descriptives.json", "w"), indent=1)
print(json.dumps(desc, indent=1)); print(loc)

# Figure: distribution of value, raw vs log
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6))
ax[0].hist(v / 1e6, bins=60, color="#1F4E79"); ax[0].set(xlabel="Value (KSh million)", ylabel="Parcels", title="Raw scale")
ax[1].hist(y, bins=40, color="#1F4E79"); ax[1].set(xlabel="log(value, KSh)", title="Log scale")
fig.tight_layout(); fig.savefig(OUT / "fig_value_distribution.png", dpi=200); plt.close(fig)

# Figure: grouped CV comparison (benchmark excluded because it would distort the scale)
cv = pd.read_csv(OUT / "cv_all_models.csv")
c = cv[~cv.Model.str.contains("Benchmark")].sort_values("MAE (KSh)")
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.1), sharey=True)
col = ["#1F4E79" if m == "Random Forest" else "#9DB4C0" for m in c.Model]
ax[0].barh(c.Model, c["MAE (KSh)"] / 1e6, xerr=c["MAE sd"] / 1e6, color=col, capsize=3)
ax[0].set_xlabel("MAE (KSh million)"); ax[0].invert_yaxis()
ax[1].barh(c.Model, c["R2 (log)"], xerr=c["R2 (log) sd"], color=col, capsize=3)
ax[1].set_xlabel("R² (log scale)")
fig.tight_layout(); fig.savefig(OUT / "fig_cv_comparison.png", dpi=200); plt.close(fig)

# Figure: permutation importance
imp = pd.read_csv(OUT / "permutation_importance.csv").sort_values("importance")
fig, ax = plt.subplots(figsize=(6.5, 2.9))
ax.barh(imp.feature, imp.importance, xerr=imp.sd, color="#1F4E79", capsize=2)
ax.set_xlabel("Increase in log-MAE when the feature is shuffled")
fig.tight_layout(); fig.savefig(OUT / "fig_permutation_importance.png", dpi=200); plt.close(fig)
