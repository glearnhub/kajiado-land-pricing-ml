"""
Supplementary check: compare all tuned models with grouped 5-fold cross-validation on the full dataset.
A single 15% test set (89 parcel blocks) is noisy, so this gives a mean and spread for every model.
Run after land_price_models.py (re-uses its data preparation and tuned hyperparameters).
"""
import ast
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

# Re-use data preparation and model definitions from the main script (sections 1-5, without tuning)
src = (HERE / "land_price_models.py").read_text(encoding="utf-8")
src = src.split("cv = GroupKFold(5)")[0]
src = src.replace('log = open(OUT / "run_log.txt", "w", encoding="utf-8")', "log = io.StringIO()")
ns = {"__file__": str(HERE / "land_price_models.py"), "io": io, "print": lambda *a, **k: None}
exec(src, ns)
X, y, groups, models, ACRES = ns["X"], ns["y"], ns["groups"], ns["models"], ns["ACRES_PER_HA"]

val = pd.read_csv(OUT / "validation_results.csv")
params = {r.Model: r["Best parameters"] for _, r in val.iterrows()}


def parse(p):
    if not isinstance(p, str) or p in ("-", "{}"):
        return {}
    return ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", p))


rows, cv = [], GroupKFold(5)
for name, (pipe, _) in list(models.items()) + [("Benchmark: section median per acre", (None, None))]:
    fold = []
    for tr, te in cv.split(X, y, groups):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        if pipe is None:
            ppa = pd.Series(ytr - np.log(np.exp(Xtr.log_area_ha) * ACRES), index=Xtr.index).groupby(Xtr.location).median()
            p = ppa.reindex(Xte.location).fillna(ppa.median()).values + np.log(np.exp(Xte.log_area_ha.values) * ACRES)
        else:
            est = clone(pipe).set_params(**{f"m__{k}": v for k, v in parse(params.get(name)).items()})
            p = est.fit(Xtr, ytr).predict(Xte)
        a, pr = np.exp(yte), np.exp(p)
        fold.append({"MAE": mean_absolute_error(a, pr), "RMSE": np.sqrt(mean_squared_error(a, pr)),
                     "MdAPE": 100 * np.median(np.abs(pr - a) / a), "R2log": r2_score(yte, p)})
    f = pd.DataFrame(fold)
    rows.append({"Model": name, "MAE (KSh)": f.MAE.mean(), "MAE sd": f.MAE.std(), "RMSE (KSh)": f.RMSE.mean(),
                 "Median abs % error": f.MdAPE.mean(), "R2 (log)": f.R2log.mean(), "R2 (log) sd": f.R2log.std()})
    print(f"{name:<38} MAE {f.MAE.mean():>12,.0f}  MdAPE {f.MdAPE.mean():5.1f}%  R2log {f.R2log.mean():.3f}")

res = pd.DataFrame(rows).sort_values("MAE (KSh)")
res.to_csv(OUT / "cv_all_models.csv", index=False)
print("\n" + res.round(3).to_string(index=False))
