"""ANN (TensorFlow/Keras) wrapped as a scikit-learn estimator, so it plugs into the shared pipeline, grid search and CV.
Overfitting is curbed by dropout, an L2 penalty and early stopping on a validation slice of the training fold."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TensorFlow start-up logs (also in GridSearchCV workers)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from constants import SEED
from models import common

NAME = "ANN"

GRID = {"clf__hidden": [(4,), (8,), (16,)],
        "clf__dropout": [0.1, 0.2, 0.3], #share of units randomly switched off each training step
        "clf__l2": [0.01, 0.03, 0.1]} # weight penalty on every dense layer

class KerasMLP(ClassifierMixin, BaseEstimator):
    """Keras network behind the sklearn interface (fit / predict_proba / get_params), so GridSearchCV,
    cross_val_predict and clone treat it like any other classifier."""

    def __init__(self, hidden=(16,), dropout=0.2, l2=1e-3, learning_rate=1e-3, epochs=300, batch_size=32,
                 patience=20, validation_fraction=0.15, random_state=SEED):
        self.hidden = hidden
        self.dropout = dropout
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X, y):
        import keras  # imported here so each GridSearchCV worker loads TensorFlow itself
        from keras import layers, regularizers

        X, y = np.asarray(X, dtype="float32"), np.asarray(y)
        self.classes_ = np.unique(y)
        keras.utils.set_random_seed(self.random_state)  # reproducible weights, dropout masks and shuffling

        # early stopping watches a slice of THIS training fold only, so the CV fold being scored is never seen
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=self.validation_fraction,
                                                    stratify=y, random_state=self.random_state)
        net = keras.Sequential([keras.Input(shape=(X.shape[1],))])
        for units in self.hidden:
            net.add(layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(self.l2)))
            net.add(layers.Dropout(self.dropout))
        net.add(layers.Dense(1, activation="sigmoid"))
        net.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="binary_crossentropy")

        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        hist = net.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=self.epochs,
                       batch_size=self.batch_size, callbacks=[stop], verbose=0)
        self.model_ = net
        self.loss_curve_ = hist.history["loss"]
        self.val_loss_curve_ = hist.history["val_loss"]
        self.best_epoch_ = int(np.argmin(self.val_loss_curve_)) + 1
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def build():
    return Pipeline([("pre", common.preprocessor()), ("clf", KerasMLP())])


def run(data):  # training data only - the test set is evaluated once in master.ipynb's Final Evaluation section
    res = common.fit(NAME, build(), GRID, data)
    model = res["model"]
    res["train_auc"] = roc_auc_score(data["y_train"], model.predict_proba(data["X_train"])[:, 1])  # for the overfitting gap
    res["loss_curve"] = model[-1].loss_curve_
    res["val_loss_curve"] = model[-1].val_loss_curve_
    res["best_epoch"] = model[-1].best_epoch_
    return res


def report(res):
    common.report(res)  # CV view
    gap = res["train_auc"] - res["cv_auc"]
    print(f"Train AUC: {res['train_auc']:.3f} | CV AUC: {res['cv_auc']:.3f} | gap: {gap:.3f}")

    plt.figure(figsize=(5, 3.5))
    plt.plot(res["loss_curve"], label="train")
    plt.plot(res["val_loss_curve"], label="validation")
    plt.axvline(res["best_epoch"] - 1, color="k", ls=":", label=f"early stop (epoch {res['best_epoch']})")
    plt.gca().set(title="Training vs validation loss", xlabel="Epoch", ylabel="Log loss + L2 penalty")
    plt.legend()
    plt.tight_layout()
    plt.show()
