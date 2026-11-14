import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from constants import SEED
from models import common

NAME = "AdaBoost"
# AdaBoost = many weak trees fitted in sequence, each focusing on the applicants the previous ones got wrong.
GRID = {
    "clf__n_estimators": [50, 100, 200, 400],
    "clf__learning_rate": [0.1, 0.3, 0.5, 1.0],                 # shrinks each tree's vote; lower needs more trees
    "clf__estimator__max_depth": [1, 2, 3],                     # 1 = stumps (no interactions), the classic weak learner
    "clf__estimator__class_weight": [None, "balanced"],         # start with bad borrowers weighted up
}


def build():
    return Pipeline([("pre", common.preprocessor()),
                     ("clf", AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=SEED),
                                                random_state=SEED))])


def run(data):  # training data only - the test set is evaluated once in master.ipynb's Final Evaluation section
    return common.fit(NAME, build(), GRID, data)


def report(res, top=15):
    common.report(res)  # CV view only
    # Weighted impurity decrease across all trees; magnitude only - see SHAP for direction.
    imp = pd.Series(res["model"]["clf"].feature_importances_,
                    index=res["model"]["pre"].get_feature_names_out()).sort_values().tail(top)
    plt.figure(figsize=(8, 0.35 * len(imp) + 1))
    plt.barh(imp.index, imp.values)
    plt.title("AdaBoost feature importance (weighted impurity decrease)\nmagnitude only - see SHAP for direction")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
