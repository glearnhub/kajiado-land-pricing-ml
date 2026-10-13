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

## Setting up on another computer

### 1. Install the tools (once per computer)

- **Python 3.12.** Download it from <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"** on the first installer screen.
- **Git.** Download it from <https://git-scm.com/downloads> and accept the default options.

To check that both are installed, open a new terminal (Command Prompt on Windows, Terminal on macOS) and run:

```bash
python --version
git --version
```

On macOS or Linux, use `python3` wherever these instructions say `python`.

### 2. Download the project

```bash
git clone https://github.com/glearnhub/kajiado-land-pricing-ml.git
cd kajiado-land-pricing-ml
```

If you'd rather not install Git, click **Code → Download ZIP** on the GitHub page. Unzip the file, then open a terminal inside the unzipped folder.

### 3. Create a virtual environment and install the packages

A virtual environment keeps this project's packages separate from anything else on the computer.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`(.venv)` should now appear at the start of the prompt. Installing the packages takes a few minutes. The versions are pinned because the saved models in `modelling/outputs/` only load with the same scikit-learn and XGBoost versions they were saved with.

### 4. Run the demo

```bash
cd modelling
python demo.py
```

The demo loads in about 10 seconds and shows a menu:

| Option | What it does |
|---|---|
| 1 | Tests all six models on the 271 held-out test parcels |
| 2 | Predicts 8 random test parcels and shows them next to their actual values |
| 3 | Prices a new parcel from details you type in (location, area, road access, ...) |
| 4 | Explains a prediction: which features raised or lowered the price |
| 0 | Exits |

### 5. Reproduce the full analysis (optional)

```bash
python land_price_models.py    # cleaning filters, grouped split, tuning, test results, SHAP (~15 minutes)
python compare_models_cv.py    # grouped 5-fold comparison of all models (~1 minute)
python report_figures.py       # figures and descriptive tables for the report
```

The results are written to `modelling/outputs/` and match the Chapter 4 tables exactly, because the runs use fixed random seeds.

### Next time

You don't need to repeat steps 1–3. Open a terminal in the project folder, activate the environment (`.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux), then run `cd modelling` and `python demo.py`.

To download the latest version of the project, run `git pull` in the project folder.

### Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized | Reinstall Python with "Add python.exe to PATH" ticked, then open a new terminal. On macOS/Linux use `python3`. |
| `activate` is blocked by PowerShell | Use Command Prompt, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once in PowerShell. |
| Error loading `final_model.joblib` or `all_models.joblib` | The package versions differ. Run `pip install -r requirements.txt` again inside the activated environment. |
| `No module named ...` | The virtual environment isn't active. Activate it (step 3) and try again. |
