import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from constants import SEED
from models import common

NAME = "Random Forest"
# n_estimators is fixed (more trees never hurts accuracy, only speed); tune how complex each tree is instead.
GRID = {
    "clf__max_depth": [2,3,4], 
    "clf__min_samples_leaf": [3,5,7],
    "clf__max_features": [0.1, "sqrt", 0.3], 
    "clf__class_weight": [None, "balanced_subsample"],
}


def build():
    return Pipeline([("pre", common.preprocessor()),
                     ("clf", RandomForestClassifier(n_estimators=500, n_jobs=1,  # GridSearchCV already parallelises
                                                    random_state=SEED))])


def run(data):  # training data only - the test set is evaluated once in master.ipynb's Final Evaluation section
    return common.fit(NAME, build(), GRID, data)


def report(res, top=15):
    common.report(res)  # CV view only
