"""Builds the single submission notebook from the three executed notebooks and the source files.
Re-run after any change:  python build_submission.py"""
import json
from pathlib import Path

OUT = "DSE4212_G6_Project1.ipynb"
SOURCES = [
    ("constants.py", "Shared constants: seed, CV folds, category labels, feature lists, protected attributes, SHAP groups."),
    ("load_data.py", "Reads `data/german.data`, maps the coded categories to readable labels and creates `default` (1 = bad borrower)."),
    ("feature_engineering.py", "Engineered features and the single stratified 80/20 train/test split."),
    ("models/common.py", "Shared workflow used by every model: preprocessing, tuning on training data only, cost-optimal threshold, "
                         "evaluation, plots, model comparison, SHAP and fairness."),
    ("models/logistic_regression.py", "Logistic regression (required by the brief)."),
    ("models/random_forest.py", "Random forest (required by the brief; the primary model)."),
    ("models/adaboost.py", "AdaBoost (required by the brief)."),
    ("models/xgboost_model.py", "XGBoost (additional model)."),
    ("models/lightGBM.py", "LightGBM (additional model)."),
    ("models/neural_network.py", "Neural network in TensorFlow/Keras, wrapped as a scikit-learn estimator (additional model)."),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def cells_of(path):
    out = []
    for c in json.load(open(path))["cells"]:
        if not "".join(c["source"]).strip():
            continue  # drop empty cells
        c.pop("id", None)
        out.append(c)
    return out


reqs = [r.strip() for r in Path("requirements.txt").read_text().split() if r.strip()]
def source_cells(paths):
    out = []
    for path in paths:
        out.append(md(f"### `{path}`\n{DESC[path]}"))
        out.append(code(f"%%writefile {path}\n{Path(path).read_text()}"))
    return out


DESC = dict(SOURCES)
NOTE = ("The first line of each code cell in this section, `%%writefile <file>`, saves the cell's code to that file "
        "so that later cells can import it. It is not part of the code itself.")
cells = [md(f"""# DSE4212/QF4212 Group Project — Project 1
## Data-Driven Credit Scoring Using Explainable Machine Learning (Group 6)

This single notebook contains all the code and results for the project, in the order the work was done.

| Part | Content |
|---|---|
| 1 | Data loading and initial EDA (full data, single variables only) |
| 2 | Feature engineering and feature EDA (training set only: features against the target) |
| 3 | Model code: the shared modelling workflow and the six models |
| 4 | Results: tuning and cross-validated comparison, final test evaluation, SHAP explainability, fairness |

**To re-run:** place this notebook next to a `data/` folder containing `german.data`, install the packages below, and run all cells from top to bottom.

Packages: {", ".join(f"`{r}`" for r in reqs)}

The saved outputs come from the group's final run. Part 4 takes several minutes to re-run because every model is tuned by grid search.""")]

cells.append(md("# Part 1 — Data loading and initial EDA\n## 1.1 Constants and data loading code\n" + NOTE))
cells += source_cells(["constants.py", "load_data.py"])
cells.append(md("## 1.2 Initial EDA"))
cells += cells_of("Initial EDA.ipynb")

cells.append(md("# Part 2 — Feature engineering and feature EDA\n## 2.1 Feature engineering code\n" + NOTE))
cells += source_cells(["feature_engineering.py"])
cells.append(md("## 2.2 Feature EDA (training set only)"))
cells += cells_of("feature EDA.ipynb")

cells.append(md("# Part 3 — Model code\n" + NOTE))
cells.append(code('import os\n\nos.makedirs("models", exist_ok=True)\nopen("models/__init__.py", "a").close()  # makes models/ an importable package'))
cells += source_cells([p for p, _ in SOURCES if p.startswith("models/")])

cells.append(md("# Part 4 — Modelling results, evaluation, explainability and fairness"))
master = cells_of("master.ipynb")
first = "".join(master[0]["source"]).replace(
    "\n\nTo add a model: copy `models/template.py`, fill in `build()` and `GRID`, then add a cell below.", "")
master[0]["source"] = first.splitlines(keepends=True)
cells += master

RENAME = {"# Initial EDA — German Credit Data": "### Initial EDA — German Credit Data",
          "# Post Feature Creation Analysis": "### Post Feature Creation Analysis",
          "# Master Notebook — Credit Default Models": "### Modelling workflow",
          "# Part 1 — Model development (training data / CV only)": "## 4.1 Model development (training data / CV only)",
          "# Part 2 — Final evaluation (run ONCE, after all models are frozen)": "## 4.2 Final evaluation (run ONCE, after all models are frozen)"}
for c in cells:  # embedded notebook titles become sub-headings so the part numbering stays consistent
    if c["cell_type"] == "markdown" and c["source"]:
        head = c["source"][0].rstrip("\n")
        if head in RENAME:
            c["source"][0] = RENAME[head] + "\n" * c["source"][0].endswith("\n")

meta = json.load(open("master.ipynb"))["metadata"]
json.dump({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 4}, open(OUT, "w"), indent=1, ensure_ascii=False)
print("wrote", OUT, "with", len(cells), "cells")
