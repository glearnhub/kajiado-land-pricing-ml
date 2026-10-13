"""
Land price models for Kajiado County (Kajiado North valuation register, 2022-2023).

Follows the methodology in Chapter 3 of the proposal
"A Standardized Machine Learning Model for Land Pricing in Kajiado County":
  Stage 2  data cleaning        -> uses the cleaned workbook + vacant-land / outlier filters
  Stage 4  feature engineering  -> size, location (section), time, accessibility text flags
  Stage 6  70 / 15 / 15 split   -> train / validation / test (test only used once, at the end).
                                   Splits and CV folds are GROUPED: neighbouring parcel numbers in the same
                                   section (blocks of 200) stay together, because the register values plots
                                   in batches and a random split would leak near-identical siblings into the test.
  3.10.3   models               -> MLR, Random Forest, SVM, Gradient Boosting, XGBoost, ANN
  3.10.4   evaluation           -> MAE, RMSE, MAPE, R2 (in KSh)
  3.10.5   tuning               -> randomized search with 5-fold cross-validation
  3.10.6   explainability       -> SHAP + permutation importance

Run:  python land_price_models.py
Outputs go to ./outputs next to this script.
"""
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, RandomizedSearchCV, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from bounded_mlp import BoundedMLP  # MLPRegressor with a standardised target and bounded predictions

warnings.filterwarnings("ignore")
SEED = 42
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Kajiado North Cleaned Data.xlsx"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
ACRES_PER_HA = 2.471
log = open(OUT / "run_log.txt", "w", encoding="utf-8")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    log.write(msg + "\n")
    log.flush()


# ---------------------------------------------------------------- 1. Load
raw = pd.read_excel(DATA, sheet_name="Cleaned Data")
raw.columns = ["year", "reg", "section", "area_raw", "ha", "value_raw", "value", "ppacre",
               "situation", "date_raw", "date", "dup", "notes"]
say(f"Records in cleaned workbook: {len(raw):,}")

# ---------------------------------------------------------------- 2. Inclusion criteria
d = raw.copy()
steps = []
d = d[d.value.notna() & d.ha.notna() & (d.ha > 0)];                  steps.append(("Has value and area", len(d)))
sit = d.situation.fillna("").astype(str)
built = sit.str.contains(r"house|\bbr\b|\d\s*br|bdrm|bedroom|apartment|\bapt|maisonette|flat|structure|building|storey|lodge", case=False) \
        | d.reg.astype(str).str.contains(r"apt|hse|house", case=False)
d = d[~built];                                                        steps.append(("Vacant land only (no buildings mentioned)", len(d)))
d = d.drop_duplicates(subset=["reg", "year", "value", "ha"]);          steps.append(("Exact duplicates removed", len(d)))
d = d[~d.section.isin(["Unknown"])];                                  steps.append(("Section known", len(d)))
lpa = np.log(d.value / (d.ha * ACRES_PER_HA))
lo, hi = lpa.quantile([0.01, 0.99])
d = d[(lpa >= lo) & (lpa <= hi)];                                     steps.append(("Price per acre within 1st-99th percentile", len(d)))
for s, n in steps:
    say(f"  {s:<45} {n:>6,}")
d = d.reset_index(drop=True)

# ---------------------------------------------------------------- 3. Feature engineering
sit = d.situation.fillna("").astype(str)
num = d.reg.astype(str).str.extract(r"(\d{2,6})")[0].astype(float)
counts = d.section.value_counts()
d["location"] = d.section.where(d.section.map(counts) >= 15, "Other")
X = pd.DataFrame({
    "log_area_ha": np.log(d.ha),
    "location": d.location,
    "year_2023": (d.year.astype(str) == "2023").astype(int),
    "dist_km": sit.str.extract(r"(\d+(?:\.\d+)?)\s*km", flags=re.I)[0].astype(float),   # distance to road/centre
    "road_access": sit.str.contains(r"road|tarmac|along|highway|\brd\b", case=False).astype(int),
    "near_amenity": sit.str.contains(r"school|town|shop|centre|center|market|church|petrol|hospital", case=False).astype(int),
    "off_road": sit.str.contains(r"\boff\b", case=False).astype(int),
    "has_description": (sit.str.len() > 0).astype(int),
    "log_parcel_no": np.log1p(num),            # higher numbers = more recent subdivisions
})
y = np.log(d.value.values)                   # models learn log(KSh); metrics reported in KSh
BLOCK = 200
groups = (d.section + "_" + (num.fillna(0) // BLOCK).astype(int).astype(str)).values
say(f"\nModelling dataset: {len(X):,} parcels, {X.shape[1]} features")
say(X.describe(include="all").T.to_string())

NUM = ["log_area_ha", "year_2023", "dist_km", "road_access", "near_amenity", "off_road", "has_description", "log_parcel_no"]
CAT = ["location"]


def prep(scale):
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale == "quantile":   # maps every input to a bounded normal scale, so outliers cannot push the network off the data
        num_steps.append(("scale", QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEED)))
    elif scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])


# ---------------------------------------------------------------- 4. Grouped split 70/15/15
say(f"Parcel blocks (groups): {len(set(groups))}")
tr_i, tmp_i = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(X, y, groups))
va_r, te_r = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED).split(tmp_i, groups=groups[tmp_i]))
va_i, te_i = tmp_i[va_r], tmp_i[te_r]
X_tr, X_va, X_te = X.iloc[tr_i], X.iloc[va_i], X.iloc[te_i]
y_tr, y_va, y_te = y[tr_i], y[va_i], y[te_i]
g_tr = groups[tr_i]
assert not (set(groups[tr_i]) & set(groups[te_i])) and not (set(groups[va_i]) & set(groups[te_i]))
say(f"\nTrain {len(X_tr)} | Validation {len(X_va)} | Test {len(X_te)}  (no parcel block shared between sets)")
pd.DataFrame({"Set": ["Train", "Validation", "Test"], "Parcels": [len(tr_i), len(va_i), len(te_i)],
              "Blocks": [len(set(groups[i])) for i in (tr_i, va_i, te_i)]}).to_csv(OUT / "split_sizes.csv", index=False)


def metrics(y_log, p_log):
    a, p = np.exp(y_log), np.exp(p_log)
    return {"MAE (KSh)": mean_absolute_error(a, p),
            "RMSE (KSh)": np.sqrt(mean_squared_error(a, p)),
            "MAPE (%)": 100 * mean_absolute_percentage_error(a, p),
            "R2 (log)": r2_score(y_log, p_log),
            "R2 (KSh)": r2_score(a, p)}


# ---------------------------------------------------------------- 5. Models and search spaces
models = {
    "Multiple Linear Regression": (Pipeline([("prep", prep(True)), ("m", LinearRegression())]), {}),
    "Random Forest": (Pipeline([("prep", prep(False)), ("m", RandomForestRegressor(random_state=SEED, n_jobs=1))]), {
        "m__n_estimators": randint(200, 800), "m__max_depth": [None, 6, 10, 16],
        "m__min_samples_leaf": randint(1, 10), "m__max_features": [0.3, 0.5, 0.7, 1.0]}),
    "Support Vector Machine": (Pipeline([("prep", prep(True)), ("m", SVR())]), {
        "m__C": loguniform(0.1, 100), "m__gamma": loguniform(1e-3, 1), "m__epsilon": uniform(0.01, 0.3)}),
    "Gradient Boosting": (Pipeline([("prep", prep(False)), ("m", GradientBoostingRegressor(random_state=SEED))]), {
        "m__n_estimators": randint(100, 800), "m__learning_rate": loguniform(0.01, 0.2),
        "m__max_depth": randint(2, 6), "m__subsample": uniform(0.6, 0.4), "m__min_samples_leaf": randint(1, 20)}),
    "XGBoost": (Pipeline([("prep", prep(False)), ("m", XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0))]), {
        "m__n_estimators": randint(100, 800), "m__learning_rate": loguniform(0.01, 0.2),
        "m__max_depth": randint(2, 8), "m__subsample": uniform(0.6, 0.4), "m__colsample_bytree": uniform(0.5, 0.5),
        "m__min_child_weight": randint(1, 10), "m__reg_lambda": loguniform(0.1, 10)}),
    "Artificial Neural Network": (Pipeline([("prep", prep("quantile")), ("m", BoundedMLP(max_iter=3000, early_stopping=True, random_state=SEED))]), {
        "m__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
        "m__activation": ["relu", "tanh"],
        "m__alpha": loguniform(1e-3, 10), "m__learning_rate_init": loguniform(1e-4, 1e-2)}),
}

cv = GroupKFold(5)
fitted, rows = {}, []

# Benchmark that mimics the comparable-sales approach: section median price per acre x area
tr_ppa = pd.Series(y_tr - np.log(np.exp(X_tr.log_area_ha) * ACRES_PER_HA), index=X_tr.index).groupby(X_tr.location).median()
bench = lambda Xs: tr_ppa.reindex(Xs.location).fillna(tr_ppa.median()).values + np.log(np.exp(Xs.log_area_ha.values) * ACRES_PER_HA)
rows.append({"Model": "Benchmark: section median per acre", "Best CV MAE log": np.nan, **metrics(y_va, bench(X_va)), "Best parameters": "-"})

for name, (pipe, space) in models.items():
    say(f"\nTuning {name} ...")
    if space:
        search = RandomizedSearchCV(pipe, space, n_iter=40, cv=cv, scoring="neg_mean_absolute_error",
                                    random_state=SEED, n_jobs=-1)
        search.fit(X_tr, y_tr, groups=g_tr)
        best, cv_mae, params = search.best_estimator_, -search.best_score_, search.best_params_
    else:
        cv_mae = -cross_val_score(pipe, X_tr, y_tr, cv=cv, groups=g_tr, scoring="neg_mean_absolute_error").mean()
        best, params = pipe.fit(X_tr, y_tr), {}
    fitted[name] = best
    m = metrics(y_va, best.predict(X_va))
    clean_params = {k.replace("m__", ""): (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()}
    rows.append({"Model": name, "Best CV MAE log": cv_mae, **m, "Best parameters": str(clean_params) or "-"})
    say(f"  validation: " + ", ".join(f"{k} {v:,.3f}" for k, v in m.items()))

val = pd.DataFrame(rows)
val.to_csv(OUT / "validation_results.csv", index=False)
say("\nVALIDATION RESULTS\n" + val.drop(columns="Best parameters").round(3).to_string(index=False))

# ---------------------------------------------------------------- 6. Select, refit on train+validation, test once
ml = val[val.Model.isin(models)].sort_values("MAE (KSh)")
best_name = ml.iloc[0].Model
say(f"\nSelected model (lowest validation MAE): {best_name}")
X_trva, y_trva = pd.concat([X_tr, X_va]), np.concatenate([y_tr, y_va])
test_rows = [{"Model": "Benchmark: section median per acre", **metrics(y_te, bench(X_te))}]
for name, est in fitted.items():
    est.fit(X_trva, y_trva)
    test_rows.append({"Model": name, **metrics(y_te, est.predict(X_te))})
test = pd.DataFrame(test_rows)
test.to_csv(OUT / "test_results.csv", index=False)
say("\nTEST RESULTS (held-out 15%, models refit on train+validation)\n" + test.round(3).to_string(index=False))
final = fitted[best_name]
joblib.dump(final, OUT / "final_model.joblib")
joblib.dump(fitted, OUT / "all_models.joblib")   # every tuned model, refit on train+validation (used by demo.py)

# Consistency: grouped 5-fold CV on all data for the selected model, and the leakage a random split would cause
cons = []
for label, splitter, kw in [("Grouped 5-fold (used)", GroupKFold(5), {"groups": groups}),
                            ("Random 5-fold (for comparison)", KFold(5, shuffle=True, random_state=SEED), {})]:
    cvp = cross_val_predict(final, X, y, cv=splitter, **kw)
    folds = [mean_absolute_error(np.exp(y[v]), np.exp(cvp[v])) for _, v in splitter.split(X, y, **kw)]
    cons.append({"Validation": label, "MAE (KSh)": np.mean(folds), "MAE fold sd (KSh)": np.std(folds),
                 "R2 (log)": r2_score(y, cvp),
                 "Median abs % error": 100 * np.median(np.abs(np.exp(cvp) - np.exp(y)) / np.exp(y))})
cons = pd.DataFrame(cons)
cons.to_csv(OUT / "cross_validation.csv", index=False)
say(f"\nCross-validation of {best_name} on all {len(X)} parcels\n" + cons.round(3).to_string(index=False))

# ---------------------------------------------------------------- 7. Charts
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
pred = final.predict(X_te)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(np.exp(y_te) / 1e6, np.exp(pred) / 1e6, s=14, alpha=0.6, color="#1F4E79")
lim = [0.01, max(np.exp(y_te).max(), np.exp(pred).max()) / 1e6 * 1.2]
ax.plot(lim, lim, color="#999999", lw=1, ls="--")
ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Actual value (KSh million)",
       ylabel="Predicted value (KSh million)", title=f"{best_name}: test set")
fig.tight_layout(); fig.savefig(OUT / "actual_vs_predicted.png", dpi=200); plt.close(fig)

t = test[test.Model.isin(models)].sort_values("MAE (KSh)")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
bars = ax.barh(t.Model, t["MAE (KSh)"] / 1e6, color=["#1F4E79" if m == best_name else "#9DB4C0" for m in t.Model])
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("Test MAE (KSh million)"); fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 8. Explainability
names = list(final.named_steps["prep"].get_feature_names_out())
names = [n.replace("num__", "").replace("cat__location_", "location: ").replace("missingindicator_", "missing: ") for n in names]
pi = permutation_importance(final, X_te, y_te, n_repeats=30, random_state=SEED, scoring="neg_mean_absolute_error")
imp = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean, "sd": pi.importances_std}).sort_values("importance", ascending=False)
imp.to_csv(OUT / "permutation_importance.csv", index=False)
say("\nPermutation importance (increase in log-MAE when feature is shuffled)\n" + imp.round(4).to_string(index=False))

try:
    import shap
    tree_name = best_name if best_name in ("Random Forest", "Gradient Boosting", "XGBoost") else \
        ml[ml.Model.isin(["Random Forest", "Gradient Boosting", "XGBoost"])].iloc[0].Model
    tm = fitted[tree_name]
    Xt = pd.DataFrame(tm.named_steps["prep"].transform(X_trva), columns=names)
    sv = shap.TreeExplainer(tm.named_steps["m"]).shap_values(Xt)
    shap.summary_plot(sv, Xt, show=False, max_display=15)
    plt.title(f"SHAP values: {tree_name} (effect on log value)", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT / "shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
    pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False) \
        .to_csv(OUT / "shap_importance.csv", index=False)
    say(f"\nSHAP computed for {tree_name}")
except ImportError:
    say("\nshap not installed - skipped SHAP plot (pip install shap)")

pd.DataFrame(steps, columns=["Step", "Records remaining"]).to_csv(OUT / "data_filtering.csv", index=False)
log.close()
