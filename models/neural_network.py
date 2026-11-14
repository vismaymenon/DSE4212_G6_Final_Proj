"""ANN (sklearn MLP). Small network; overfitting is curbed by an L2 penalty, a CV tuned iteration cap and low capacity."""
import warnings

import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from constants import SEED
from models import common

NAME = "ANN"
GRID = {"clf__hidden_layer_sizes": [(4,), (8,), (16,), (8, 4)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [0.01, 0.1, 1, 3, 10],      # L2 penalty, divided by n inside sklearn so useful values are large
        "clf__max_iter": [50, 100, 200, 400]}     # iteration cap acts as early stopping, chosen by CV AUC


class QuietMLP(MLPClassifier):
    """MLPClassifier that hides ConvergenceWarning (stopping at max_iter is intended). Needed because
    n_jobs=-1 workers do not inherit the notebook's warning filters."""
    def fit(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return super().fit(*args, **kwargs)


def build():
    return Pipeline([("pre", common.preprocessor()),
                     ("clf", QuietMLP(solver="adam", learning_rate_init=0.001, batch_size=32, random_state=SEED))])


def run(data):  # training data only - the test set is evaluated once in master.ipynb's Final Evaluation section
    res = common.fit(NAME, build(), GRID, data)
    model = res["model"]
    res["train_auc"] = roc_auc_score(data["y_train"], model.predict_proba(data["X_train"])[:, 1])  # for the overfitting gap
    res["loss_curve"] = model[-1].loss_curve_
    return res


def report(res):
    common.report(res)  # CV view
    gap = res["train_auc"] - res["cv_auc"]
    print(f"Train AUC: {res['train_auc']:.3f} | CV AUC: {res['cv_auc']:.3f} | gap: {gap:.3f}")

    plt.figure(figsize=(5, 3.5))
    plt.plot(res["loss_curve"])
    plt.gca().set(title=f"Training loss (max_iter={res['best_params']['clf__max_iter']})",
                  xlabel="Epoch", ylabel="Log loss + L2 penalty")
    plt.tight_layout()
    plt.show()