# DSE4212_G6_Final_Proj

Project 1: credit default prediction on the German Credit data (UCI Statlog).

## Structure
```
Initial EDA.ipynb        full data, univariate only (distributions, skew, category counts, coding checks); never relates features to the target
feature EDA.ipynb        TRAIN set only: feature vs target (bad rates, correlation, VIF, Information Value with bootstrap CIs)
master.ipynb             Part 1: one cell per model + CV comparison. Part 2: final test evaluation, SHAP, fairness

constants.py             seed, CV folds, column labels, feature lists (NUMERIC / ORDINAL / CATEGORICAL / BINARY), PROTECTED
load_data.py             reads data/german.data, maps codes to labels, default = 1 for bad borrowers
feature_engineering.py   engineered features + the single stratified 80/20 train/test split (get_data)

models/common.py         shared preprocessor, fit (train/CV only), evaluate (test set), metrics, plots, compare, SHAP, fairness
models/logistic_regression.py, random_forest.py, adaboost.py      models required by the brief
models/xgboost_model.py, lightGBM.py, neural_network.py           additional models
models/template.py       copy this to add a model
```

## Setup
`python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

Use the venv. scikit-learn >= 1.8 is required: older versions silently ignore `l1_ratio`, so logistic regression never tries Lasso / elastic net.

## Adding a model
Copy `models/template.py`, set `NAME`, `build()` and `GRID` (parameter names prefixed with `clf__`), then add a cell in Part 1 of `master.ipynb`.
Every model uses the same split, CV folds, preprocessor and cost matrix, so results are directly comparable.

## Key decisions
- Cost matrix from the dataset documentation: approving a bad borrower costs 5, rejecting a good one costs 1. Each model's decision threshold minimises this cost on out-of-fold training predictions.
- `sex` and `foreign_worker` are never model inputs; they are used only in the fairness analysis. Both have known coding problems (Grömping 2019, "South German Credit"): `foreign_worker` is reversed and sex cannot be fully recovered from `personal_status`. Fairness results are indicative only.
- `age` is a model input.
- No income field exists, so loan-to-income is approximated (monthly payment = amount / duration; `installment_rate` is the instalment as a % of disposable income).

## Test-set discipline
`run()` only ever sees training data. Compare and choose models on CV results (`common.compare(..., on="cv")`).
The test set is scored once, in Part 2 of `master.ipynb`, after every model is frozen. Don't tweak models after looking at test results.

Notebooks are for outward-facing results; `.py` files hold code that is called.
