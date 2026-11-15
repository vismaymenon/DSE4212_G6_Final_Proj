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
    "clf__l1_ratio": [0.2, 0.5, 0.8, 1.0],
    "clf__class_weight": [None, "balanced"],  # None = no class weighting, "balanced" = inverse frequency of classes (Bad is weighted more heavily)
}


def build():
    return Pipeline([("pre", common.preprocessor()),
                     ("clf", LogisticRegression(solver="saga", max_iter=5000, random_state=SEED))]) #saga solver supports L1, L2, and elastic net regularization. 


def run(data):
    return common.fit(NAME, build(), GRID, data)


def report(res, top=15):
    common.report(res)  # CV view only; test-set results come from the Final Evaluation section
