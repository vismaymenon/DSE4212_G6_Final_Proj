"""LightGBM credit-default model using the shared training workflow."""

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from constants import SEED
from models import common


NAME = "LightGBM"


def build():
    return Pipeline([
        ("pre", common.preprocessor()),
        ("clf", LGBMClassifier(
            objective="binary",
            random_state=SEED,
            n_jobs=1,
            verbosity=-1,
            importance_type="gain",
        )),
    ])


# 32 combinations, assessed using the shared 5-fold training-only CV.
GRID = {
    "clf__num_leaves": [7, 15],
    "clf__min_child_samples": [20, 40],
    "clf__learning_rate": [0.03, 0.10],
    "clf__n_estimators": [100, 250],
    "clf__reg_lambda": [0.0, 5.0],
}


def run(data):
    return common.fit(NAME, build(), GRID, data)


def report(res, on="cv"):
    common.report(res, on=on)

    # Importance comes from the selected model fitted on TRAIN, not TEST.
    model = res["model"]
    names = model.named_steps["pre"].get_feature_names_out()
    gains = model.named_steps["clf"].feature_importances_
    importance = pd.Series(gains, index=names).sort_values(ascending=False)

    if importance.sum() == 0:
        print("LightGBM did not make any feature splits; no importances to plot.")
        return

    top = (100 * importance.head(15) / importance.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot.barh(ax=ax)
    ax.set_title("LightGBM: Top 15 features by split gain")
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_ylabel("")
    fig.tight_layout()
    plt.show()
