import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_predict

from features import CV, SEED

COST_FN, COST_FP = 5, 1  # german.doc cost matrix: bad classed good = 5, good classed bad = 1
THRESHOLDS = np.linspace(0.01, 0.99, 99)


def cost(y, pred):
    _, fp, fn, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return COST_FN * fn + COST_FP * fp


def metrics(y, proba, t):
    pred = (proba >= t).astype(int)
    return {"auc": roc_auc_score(y, proba), "avg_precision": average_precision_score(y, proba),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "reject_rate": pred.mean(), "cost_per_applicant": cost(y, pred) / len(y)}


def tune(name, pipe, grid, data, scoring="roc_auc"):
    search = GridSearchCV(pipe, grid, scoring=scoring, cv=CV, n_jobs=-1)
    search.fit(data["X_train"], data["y_train"])
    model = search.best_estimator_

    # threshold picked on out-of-fold train predictions so the test set is never used for tuning
    oof = cross_val_predict(clone(model), data["X_train"], data["y_train"], cv=CV, method="predict_proba")[:, 1]
    threshold = THRESHOLDS[np.argmin([cost(data["y_train"], oof >= t) for t in THRESHOLDS])]

    proba = model.predict_proba(data["X_test"])[:, 1]
    y = data["y_test"].to_numpy()
    return {"name": name, "model": model, "best_params": search.best_params_,
            "cv_auc": search.best_score_, "cv_auc_std": search.cv_results_["std_test_score"][search.best_index_],
            "threshold": threshold, "proba": proba, "y_test": y,
            "metrics": pd.DataFrame({"t=0.50": metrics(y, proba, 0.5),
                                     f"t={threshold:.2f} (cost-opt)": metrics(y, proba, threshold)}).T}


def report(res):
    y, p, t = res["y_test"], res["proba"], res["threshold"]
    print(f"{res['name']}\nBest params: {res['best_params']}\n"
          f"CV AUC: {res['cv_auc']:.3f} ± {res['cv_auc_std']:.3f}   Test AUC: {roc_auc_score(y, p):.3f}")
    display(res["metrics"].round(3))

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC", xlabel="FPR", ylabel="TPR")
    prec, rec, _ = precision_recall_curve(y, p)
    ax[1].plot(rec, prec)
    ax[1].axhline(y.mean(), color="k", ls="--", lw=0.8)
    ax[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    ConfusionMatrixDisplay.from_predictions(y, (p >= t).astype(int), display_labels=["good", "bad"],
                                            ax=ax[2], colorbar=False)
    ax[2].set_title(f"Confusion matrix (t={t:.2f})")
    ax[3].plot(THRESHOLDS, [cost(y, p >= th) / len(y) for th in THRESHOLDS])
    ax[3].axvline(t, color="r", ls="--", label="chosen (CV)")
    ax[3].axvline(0.5, color="k", ls=":", label="0.5")
    ax[3].set(title="Expected cost vs threshold (test)", xlabel="Threshold", ylabel="Cost per applicant")
    ax[3].legend()
    plt.tight_layout()
    plt.show()


def compare(results):
    results = list(results)
    rows = {r["name"]: {"cv_auc": r["cv_auc"], "cv_auc_std": r["cv_auc_std"], "threshold": r["threshold"],
                        **metrics(r["y_test"], r["proba"], r["threshold"])} for r in results}
    y = results[0]["y_test"]
    rows["Approve all"] = {"cost_per_applicant": COST_FN * y.mean(), "reject_rate": 0.0}
    rows["Reject all"] = {"cost_per_applicant": COST_FP * (1 - y.mean()), "reject_rate": 1.0}
    display(pd.DataFrame(rows).T.round(3))

    for r in results:
        fpr, tpr, _ = roc_curve(r["y_test"], r["proba"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC={roc_auc_score(r['y_test'], r['proba']):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.gca().set(title="ROC comparison (test)", xlabel="FPR", ylabel="TPR")
    plt.legend()
    plt.show()


def explain(res, data, max_display=15):
    import shap

    pre, clf = res["model"][:-1], res["model"][-1]
    names = pre.get_feature_names_out()
    X_bg = pd.DataFrame(pre.transform(data["X_train"]), columns=names)
    X_te = pd.DataFrame(pre.transform(data["X_test"]), columns=names)
    try:
        explainer = shap.Explainer(clf, X_bg)  # exact linear/tree explainer when supported
    except Exception:
        explainer = shap.Explainer(lambda x: clf.predict_proba(x)[:, 1], shap.sample(X_bg, 100, random_state=SEED))
    sv = explainer(X_te)
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]

    shap.plots.bar(sv, max_display=max_display)
    shap.plots.beeswarm(sv, max_display=max_display)
    worst = int(np.argmax(res["proba"]))
    print(f"Highest-risk test applicant: P(default) = {res['proba'][worst]:.2f}")
    shap.plots.waterfall(sv[worst], max_display=max_display)
    return sv


def fairness(res, data):
    g = data["groups_test"].reset_index(drop=True)
    y, p = res["y_test"], res["proba"]
    pred = (p >= res["threshold"]).astype(int)
    rows = []
    for attr in g.columns:
        for val, idx in g.groupby(attr).groups.items():
            yi, pi = y[idx], pred[idx]
            rows.append({"attribute": attr, "group": val, "n": len(idx), "default_rate": yi.mean(),
                         "approval_rate": 1 - pi.mean(),
                         "tpr (bad caught)": pi[yi == 1].mean() if yi.sum() else np.nan,
                         "fpr (good rejected)": pi[yi == 0].mean() if (yi == 0).sum() else np.nan,
                         "auc": roc_auc_score(yi, p[idx]) if 0 < yi.sum() < len(yi) else np.nan})
    out = pd.DataFrame(rows).set_index(["attribute", "group"])
    # disparate impact: group approval rate / best-treated group's approval rate (< 0.8 is the usual red flag)
    out["approval_ratio"] = out["approval_rate"] / out.groupby(level=0)["approval_rate"].transform("max")
    display(out.round(3))
    return out
