import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from constants import SEED
from models import common

NAME = "Random Forest"
# n_estimators is fixed (more trees never hurts accuracy, only speed); tune how complex each tree is instead.
GRID = {
    "clf__max_depth": [4, 8, None],                         # None = grow until leaves are pure / hit min_samples_leaf
    "clf__min_samples_leaf": [1, 5, 10],                    # larger leaves = smoother probabilities, less overfitting
    "clf__max_features": [0.1, "sqrt", 0.3],                # share of the 39 columns tried per split (sqrt ~ 6); lower = more decorrelated trees
    "clf__class_weight": [None, "balanced_subsample"],      # reweights bad borrowers within each bootstrap sample
}


def build():
    return Pipeline([("pre", common.preprocessor()),
                     ("clf", RandomForestClassifier(n_estimators=500, n_jobs=1,  # GridSearchCV already parallelises
                                                    random_state=SEED))])


def run(data):  # training data only - the test set is evaluated once in master.ipynb's Final Evaluation section
    return common.fit(NAME, build(), GRID, data)


def report(res, top=15):
    common.report(res)  # CV view only
    # Impurity (MDI) importance: quick, but favours continuous features with many split points - see SHAP for direction.
    imp = pd.Series(res["model"]["clf"].feature_importances_,
                    index=res["model"]["pre"].get_feature_names_out()).sort_values().tail(top)
    plt.figure(figsize=(8, 0.35 * len(imp) + 1))
    plt.barh(imp.index, imp.values)
    plt.title("Random Forest feature importance (mean impurity decrease)\nmagnitude only - see SHAP for direction")
    plt.xlabel("Mean decrease in impurity")
    plt.tight_layout()
    plt.show()
