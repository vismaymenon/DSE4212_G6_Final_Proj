import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from constants import SEED
from models import common

NAME = "Logistic Regression"
# Hyperparameter grid for logistic regression: regularization strength (C), l1_ratio for elastic net, and class weight.
GRID = {
    "clf__C": np.logspace(-3, 2, 11), # Regularisation strength (inverse of regularisation parameter). Smaller values specify stronger regularization.
    "clf__l1_ratio": [0.0, 0.5, 1.0],  # 0 = ridge, 0.5 = elastic net, 1 = lasso
    "clf__class_weight": [None, "balanced"],  # None = no class weighting, "balanced" = inverse frequency of classes (Bad is weighted more heavily)
}


def build():
    return Pipeline([("pre", common.preprocessor()),
                     ("clf", LogisticRegression(solver="saga", max_iter=5000, random_state=SEED))]) #saga solver supports L1, L2, and elastic net regularization. 


def run(data):
    return common.fit(NAME, build(), GRID, data)


def report(res, top=15):
    common.report(res)  # CV view only; test-set results come from the Final Evaluation section
    coef = pd.Series(res["model"]["clf"].coef_[0], index=res["model"]["pre"].get_feature_names_out())
    coef = coef[coef != 0]
    shown = coef.reindex(coef.abs().sort_values().index[-top:])
    odds = np.exp(shown)
    plt.figure(figsize=(8, 0.35 * len(odds) + 1))
    plt.barh(odds.index, odds - 1, left=1, color=np.where(odds > 1, "tab:red", "tab:green"))
    plt.axvline(1, color="k", lw=0.8)
    plt.title("Odds ratios (per 1 SD for numeric; vs. not-in-category for dummies; per unit for ordinal and binary)\nred = raises default risk")
    plt.xlabel("Odds ratio")
    plt.tight_layout()
    plt.show()
    print(f"{len(coef)} of {len(res['model']['clf'].coef_[0])} features kept (non-zero)")
