# A Standardized Machine Learning Model for Land Pricing in Kajiado County

Machine learning models for land prices in Kajiado North, Kenya, built from the 2022–2023 valuation register.
Six algorithms are compared, as the research proposal specifies: Multiple Linear Regression, Random Forest, SVM, Gradient Boosting, XGBoost and an Artificial Neural Network. Validation is grouped and the final model is explained with SHAP.

## Results (grouped 5-fold cross-validation, 1,410 vacant parcels)

| Model | MAE (KSh) | R² (log) |
|---|---|---|
| **Random Forest (selected)** | **1.98M** | **0.43** |
| Gradient Boosting | 2.00M | 0.47 |
| XGBoost | 2.04M | 0.42 |
| Artificial Neural Network | 2.14M | 0.32 |
| Support Vector Machine | 2.35M | 0.39 |
| Multiple Linear Regression | 2.36M | 0.34 |
| Benchmark: section median price per acre | 13.85M | −0.06 |

The register values neighbouring parcels in batches. All splits and folds therefore keep blocks of 200 consecutive parcel numbers in the same set. A random split would overstate R² (0.61 vs 0.41).

## Contents

| Path | Description |
|---|---|
| `Kajiado North 2022-2025.xlsx` | Original valuation register |
| `Kajiado North Cleaned Data.xlsx` | Cleaned data with standardised sections, value per acre and cleaning notes |
| `Kajiado North Valuation Report.docx` | Descriptive analysis of the register |
| `Chapter 4 - Results and Discussion.docx` | Draft results chapter |
| `modelling/land_price_models.py` | Main pipeline: filtering, features, grouped 70/15/15 split, tuning, testing, SHAP |
| `modelling/compare_models_cv.py` | Grouped 5-fold comparison of all tuned models |
| `modelling/report_figures.py` | Figures and descriptive tables for the report |
| `modelling/bounded_mlp.py` | Neural network with standardised target and bounded predictions |
| `modelling/demo.py` | Interactive demo: test models, predict parcels, explain predictions |
| `modelling/outputs/` | Results tables, charts and saved models |

## Running

```bash
pip install -r requirements.txt
cd modelling
python demo.py                 # live demo using the saved models (under a minute)
python land_price_models.py    # full retraining and tuning (~15 minutes)
python compare_models_cv.py
python report_figures.py
```
