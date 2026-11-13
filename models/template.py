"""Copy this file to add a model. Every model module exposes NAME, build(), run(data), report(res)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from features import SEED, preprocessor
from models import common

NAME = "Model name"
GRID = {"clf__n_estimators": [200, 500]}  # keys must be prefixed with "clf__"


def build():
    return Pipeline([("pre", preprocessor()), ("clf", RandomForestClassifier(random_state=SEED))])


def run(data):
    return common.tune(NAME, build(), GRID, data)


def report(res):
    common.report(res)  # add model-specific plots (e.g. feature importances) below
