"""LightGBM credit-default model using the shared training workflow."""

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
        )),
    ])


# 32 combinations, assessed using the shared 5-fold training-only CV.
GRID = {
    "clf__num_leaves": [3, 7, 15],
    "clf__min_child_samples": [20, 40],
    "clf__learning_rate": [0.03, 0.10],
    "clf__n_estimators": [100, 200],
    "clf__colsample_bytree": [0.5, 1.0],
    "clf__reg_lambda": [0.0, 5.0],
}


def run(data):
    return common.fit(NAME, build(), GRID, data)


def report(res, on="cv"):
    common.report(res, on=on)