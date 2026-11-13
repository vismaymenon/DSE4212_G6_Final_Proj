import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import SEED, preprocessor
from models import common

NAME = "Logistic Regression"
GRID = {
    "clf__C": np.logspace(-3, 2, 11),
    "clf__l1_ratio": [0.0, 0.5, 1.0],  # 0 = ridge, 1 = lasso (sklearn >= 1.8 API)
    "clf__class_weight": [None, "balanced"],
}


def build():
    return Pipeline([("pre", preprocessor()),
                     ("clf", LogisticRegression(solver="saga", max_iter=5000, random_state=SEED))])


def run(data):
    return common.tune(NAME, build(), GRID, data)


def report(res, top=15):
    common.report(res)
    coef = pd.Series(res["model"]["clf"].coef_[0], index=res["model"]["pre"].get_feature_names_out())
    coef = coef[coef != 0]
    shown = coef.reindex(coef.abs().sort_values().index[-top:])
    odds = np.exp(shown)
    plt.figure(figsize=(8, 0.35 * len(odds) + 1))
    plt.barh(odds.index, odds - 1, left=1, color=np.where(odds > 1, "tab:red", "tab:green"))
    plt.axvline(1, color="k", lw=0.8)
    plt.title("Odds ratios (per 1 SD for numeric; vs. not-in-category for dummies)\nred = raises default risk")
    plt.xlabel("Odds ratio")
    plt.tight_layout()
    plt.show()
    print(f"{len(coef)} of {len(res['model']['clf'].coef_[0])} features kept (non-zero)")
