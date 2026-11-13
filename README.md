# DSE4212_G6_Final_Proj

Project 1 — credit default prediction on the German Credit data.

```
EDA.ipynb          EDA + feature engineering walkthrough
master.ipynb       one cell per model, plus comparison / SHAP / fairness
features.py        data loading, engineered features, preprocessing, train/test split, CV folds
models/common.py   shared fit (train/CV only), evaluate (test set), metrics, plots, SHAP, fairness
models/logistic_regression.py
models/template.py copy this to add a model (Random Forest, AdaBoost)
```

Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

Adding a model: copy `models/template.py` to e.g. `models/random_forest.py`, set `NAME`, `build()` and `GRID`
(param names prefixed with `clf__`), then uncomment its cell in `master.ipynb`. All models share the same split,
CV folds and cost matrix, so results are directly comparable.

Test-set discipline: `run()` only ever sees training data. Compare and choose models on CV results
(`common.compare(..., on="cv")`). The test set is scored once, in the "Final evaluation" section of `master.ipynb`,
after every model is frozen. Don't tweak models after looking at test results.
