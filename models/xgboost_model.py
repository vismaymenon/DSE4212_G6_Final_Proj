import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import SEED, preprocessor
from models import common

NAME = "XGBoost"
# Small data (800 training rows) -> favour shallow trees, slow learning and subsampling to limit overfitting.
GRID = {
    "clf__n_estimators": [300, 600],
    "clf__learning_rate": [0.01, 0.03],      # small steps + more trees generalise better on small data
    "clf__max_depth": [3, 4, 6],             # tree depth = order of feature interactions allowed
    "clf__min_child_weight": [1, 3],         # min (hessian) weight per leaf; higher = more conservative splits
    "clf__colsample_bytree": [0.3, 0.5],     # fraction of features sampled per tree (decorrelates trees)
    "clf__scale_pos_weight": [1, 7 / 3],     # 7/3 = good/bad ratio, XGBoost's analogue of class_weight="balanced"
}


def build():
    return Pipeline([("pre", preprocessor()),
                     ("clf", XGBClassifier(subsample=0.8, reg_lambda=1.0, tree_method="hist",
                                           eval_metric="logloss", n_jobs=1,  # GridSearchCV already parallelises
                                           random_state=SEED))])


def run(data):  # training data only - the test set is evaluated once in master.ipynb's Final Evaluation section
    return common.fit(NAME, build(), GRID, data)


def report(res, top=15):
    common.report(res)  # CV view only
    # Gain = average loss reduction from splits on a feature; more informative than split counts.
    gain = res["model"]["clf"].get_booster().get_score(importance_type="gain")
    names = res["model"]["pre"].get_feature_names_out()
    imp = pd.Series({names[int(k[1:])]: v for k, v in gain.items()}).sort_values().tail(top)  # keys are "f0", "f1", ...
    plt.figure(figsize=(8, 0.35 * len(imp) + 1))
    plt.barh(imp.index, imp.values)
    plt.title("XGBoost feature importance (gain)\nmagnitude only - see SHAP for direction")
    plt.xlabel("Average gain per split")
    plt.tight_layout()
    plt.show()
