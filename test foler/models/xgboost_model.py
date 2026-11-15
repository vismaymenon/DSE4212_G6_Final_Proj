import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from constants import SEED
from models import common

NAME = "XGBoost"
# Small data (800 training rows) -> favour shallow trees, slow learning and subsampling to limit overfitting.
GRID = {
    "clf__n_estimators": [150, 300],
    "clf__learning_rate": [0.01, 0.03],      # small steps + more trees generalise better on small data
    "clf__max_depth": [2, 3,4],             # tree depth = order of feature interactions allowed
    "clf__min_child_weight": [1, 3,5],         # min (hessian) weight per leaf; higher = more conservative splits
    "clf__colsample_bytree": [0.3, 0.5],     # fraction of features sampled per tree (decorrelates trees)
    "clf__scale_pos_weight": [1, 7 / 3],     # 7/3 = good/bad ratio, XGBoost's analogue of class_weight="balanced"
}


def build():
    return Pipeline([("pre", common.preprocessor()),
                     ("clf", XGBClassifier(subsample=0.8, reg_lambda=1.0, tree_method="hist",
                                           eval_metric="logloss", n_jobs=1,  # GridSearchCV already parallelises
                                           random_state=SEED))])


def run(data):  # training data only - the test set is evaluated once in master.ipynb's Final Evaluation section
    return common.fit(NAME, build(), GRID, data)


def report(res, top=15):
    common.report(res)  # CV view only
