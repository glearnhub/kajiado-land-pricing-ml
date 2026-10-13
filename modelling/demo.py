"""
Live demonstration of the land price models (runs in under a minute, no re-tuning).

    python demo.py

Menu:
  1  Test all six models on the held-out test set (the 15% of parcel blocks never used in training)
  2  Show predictions for random test parcels, next to their real assessed values
  3  Price a new parcel - type in its location, size and description
  4  Explain one prediction (which features pushed the price up or down)
  5  Open the report charts (Figures 4.1-4.5) in the image viewer
  0  Exit
"""
import io
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
pd.set_option("display.width", 140)

print("Loading data and models ...")
src = (HERE / "land_price_models.py").read_text(encoding="utf-8").split("cv = GroupKFold(5)")[0]
src = src.replace('log = open(OUT / "run_log.txt", "w", encoding="utf-8")', "log = io.StringIO()")
ns = {"__file__": str(HERE / "land_price_models.py"), "io": io, "print": lambda *a, **k: None}
exec(src, ns)  # same cleaning, features and grouped 70/15/15 split as the main script
X, y, d, models = ns["X"], ns["y"], ns["d"], ns["models"]
X_tr, X_va, X_te, y_tr, y_va, y_te = (ns[k] for k in ["X_tr", "X_va", "X_te", "y_tr", "y_va", "y_te"])
te_i = ns["te_i"]
X_trva, y_trva = pd.concat([X_tr, X_va]), np.concatenate([y_tr, y_va])
final = joblib.load(OUT / "final_model.joblib")          # selected model: Random Forest
LOCATIONS = sorted(X.location.unique())


def ksh(v):
    return f"KSh {v:,.0f}"


def rf_range(Xrow):
    """Spread of the 401 trees' predictions: 10th-90th percentile, in KSh."""
    Z = final.named_steps["prep"].transform(Xrow)
    trees = np.array([t.predict(Z) for t in final.named_steps["m"].estimators_])[:, 0]
    return np.exp(np.percentile(trees, 10)), np.exp(np.percentile(trees, 90))


def test_all():
    saved = OUT / "all_models.joblib"
    if not saved.exists():
        print("  outputs/all_models.joblib not found - run land_price_models.py first.")
        return
    fitted = joblib.load(saved)   # the exact tuned models from the main run, trained on train+validation only
    rows = []
    for name, est in fitted.items():
        p = est.predict(X_te)
        a, pr = np.exp(y_te), np.exp(p)
        rows.append({"Model": name, "MAE (KSh)": f"{mean_absolute_error(a, pr):,.0f}",
                     "RMSE (KSh)": f"{np.sqrt(mean_squared_error(a, pr)):,.0f}",
                     "MAPE": f"{100 * mean_absolute_percentage_error(a, pr):.0f}%",
                     "Median error": f"{100 * np.median(np.abs(pr - a) / a):.0f}%",
                     "R2 (log)": f"{r2_score(y_te, p):.3f}"})
    print(f"\nTEST SET: {len(X_te)} parcels the models never saw (no parcel block shared with training)\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nModels were tuned with grouped 5-fold cross-validation and fitted on train+validation (land_price_models.py).")
    print("Median error = half of all predictions are within this % of the real value.")


def sample(n=8):
    idx = np.random.default_rng().choice(len(X_te), n, replace=False)
    Xs = X_te.iloc[idx]
    pred = np.exp(final.predict(Xs))
    rows = []
    for j, i in enumerate(idx):
        lo, hi = rf_range(Xs.iloc[[j]])
        actual = np.exp(y_te[i])
        rows.append({"Parcel": d.reg.iloc[te_i[i]], "Area (ha)": round(d.ha.iloc[te_i[i]], 4),
                     "Actual (KSh)": f"{actual:,.0f}", "Predicted (KSh)": f"{pred[j]:,.0f}",
                     "Error": f"{100 * (pred[j] - actual) / actual:+.0f}%",
                     "80% range": f"{lo / 1e6:.2f}M - {hi / 1e6:.2f}M",
                     "Actual in range": "yes" if lo <= actual <= hi else "no"})
    print("\nRandom Forest on randomly chosen TEST parcels:\n")
    print(pd.DataFrame(rows).to_string(index=False))


def ask(prompt, cast=str, default=None):
    v = input(prompt).strip()
    if not v:
        return default
    try:
        return cast(v)
    except ValueError:
        print("  (not understood, using default)")
        return default


def new_parcel():
    print("\nLocations:", ", ".join(f"{i + 1}={l}" for i, l in enumerate(LOCATIONS)))
    k = ask("Location number: ", int, LOCATIONS.index("Ngong") + 1)
    loc = LOCATIONS[min(max(k, 1), len(LOCATIONS)) - 1]
    unit = (ask("Area unit - h for hectares, a for acres [a]: ", str, "a") or "a").lower()
    area = ask("Area: ", float, 0.125 if unit.startswith("a") else 0.05)
    ha = area / 2.471 if unit.startswith("a") else area
    year = ask("Valuation year 2022/2023 [2023]: ", int, 2023)
    road = ask("On or along a road? y/n [n]: ", str, "n").lower().startswith("y")
    amen = ask("Near school/town/shop/centre? y/n [n]: ", str, "n").lower().startswith("y")
    dist = ask("Distance to road/centre in km (blank if unknown): ", float, None)
    pno = ask("Parcel number (blank if unknown): ", float, None)
    row = pd.DataFrame([{"log_area_ha": np.log(ha), "location": loc, "year_2023": int(year == 2023),
                         "dist_km": dist if dist is not None else np.nan, "road_access": int(road),
                         "near_amenity": int(amen), "off_road": 0, "has_description": int(road or amen or dist is not None),
                         "log_parcel_no": np.log1p(pno) if pno else np.nan}])
    p = np.exp(final.predict(row)[0]); lo, hi = rf_range(row)
    print(f"\n  {loc}, {ha:.4f} ha ({ha * 2.471:.3f} acres), {year}")
    print(f"  Predicted value : {ksh(p)}")
    print(f"  80% range       : {ksh(lo)} - {ksh(hi)}")
    print(f"  Per acre        : {ksh(p / (ha * 2.471))}")
    if not (X.log_area_ha.min() <= np.log(ha) <= X.log_area_ha.max()):
        print("  Note: this area is outside the sizes in the training data, treat with caution.")
    return row


def explain(row=None):
    import shap
    if row is None:
        i = np.random.default_rng().integers(len(X_te)); row = X_te.iloc[[i]]
        print(f"\nTest parcel {d.reg.iloc[te_i[i]]}: actual {ksh(np.exp(y_te[i]))}")
    prep, rf = final.named_steps["prep"], final.named_steps["m"]
    raw_names = list(prep.get_feature_names_out())
    Z = prep.transform(row)
    r = row.iloc[0]

    def label(n, z):   # plain-English description of each feature for this parcel
        if n.startswith("cat__location_"):
            loc = n.replace("cat__location_", "")
            return f"Located in {loc}" if z == 1 else f"Not in {loc}"
        n = n.replace("num__", "")
        if n == "log_area_ha":
            return f"Parcel size ({np.exp(r.log_area_ha):.3f} ha)"
        if n == "year_2023":
            return "Valued in 2023" if r.year_2023 else "Valued in 2022"
        if n == "log_parcel_no":
            return "Parcel number" + (f" ({np.expm1(r.log_parcel_no):.0f})" if pd.notna(r.log_parcel_no) else " (unknown)")
        if n == "road_access":
            return "On / along a road" if r.road_access else "No road mentioned"
        if n == "near_amenity":
            return "Near school/town/shop" if r.near_amenity else "No amenity mentioned"
        if n == "has_description":
            return "Has a location description" if r.has_description else "No location description"
        if n == "dist_km":
            return f"Distance {r.dist_km:g} km" if pd.notna(r.dist_km) else "Distance (unknown)"
        if n.startswith("missingindicator_"):
            return n.replace("missingindicator_", "").replace("dist_km", "Distance").replace("log_parcel_no", "Parcel number") + (" not recorded" if z == 1 else " recorded")
        if n == "off_road":
            return "Off the road" if r.off_road else "Not described as off-road"
        return n

    ex = shap.TreeExplainer(rf)
    sv = ex.shap_values(Z)[0]
    base = float(np.ravel(ex.expected_value)[0])
    print(f"\n  Starting point - typical parcel in the data: {ksh(np.exp(base))}")
    for k in np.argsort(-np.abs(sv))[:6]:
        print(f"  {label(raw_names[k], Z[0, k]):<34} {'raises' if sv[k] > 0 else 'lowers'} the price by {abs(np.exp(sv[k]) - 1) * 100:4.0f}%")
    print(f"  = Prediction: {ksh(np.exp(base + sv.sum()))}   (remaining features have small effects)")


CHARTS = [
    ("fig_value_distribution.png", "Distribution of land values, raw and log scale (Figure 4.1)"),
    ("fig_cv_comparison.png", "Model comparison: grouped cross-validation MAE and R2 (Figure 4.2)"),
    ("actual_vs_predicted.png", "Random Forest: actual vs predicted on the test set (Figure 4.3)"),
    ("fig_permutation_importance.png", "Permutation importance (Figure 4.4)"),
    ("shap_summary.png", "SHAP summary: how each feature moves the price (Figure 4.5)"),
]


def open_file(path):
    import os, subprocess, sys
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def charts():
    print()
    for i, (f, title) in enumerate(CHARTS, 1):
        print(f"  {i}  {title}")
    print("  A  Open all")
    k = input("Which chart? ").strip().lower()
    pick = CHARTS if k == "a" else [CHARTS[int(k) - 1]] if k.isdigit() and 1 <= int(k) <= len(CHARTS) else []
    for f, title in pick:
        path = OUT / f
        if path.exists():
            open_file(path)
            print(f"  opened: {title}")
        else:
            print(f"  missing {f} - run land_price_models.py and report_figures.py first")


last = None
while True:
    print("\n=== Kajiado land price model demo ===")
    print("1 Test all models on held-out data   2 Predict random test parcels")
    print("3 Price a new parcel                 4 Explain a prediction")
    print("5 Show charts                        0 Exit")
    c = input("Choose: ").strip()
    if c == "1": test_all()
    elif c == "2": sample()
    elif c == "3": last = new_parcel()
    elif c == "4": explain(last if last is not None and ask("Explain the parcel you just priced? y/n [y]: ", str, "y").lower().startswith("y") else None)
    elif c == "5": charts()
    elif c == "0": break
