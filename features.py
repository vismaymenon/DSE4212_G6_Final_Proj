from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA = Path(__file__).resolve().parent / "data" / "german.data"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

COLUMNS = ["checking", "duration", "history", "purpose", "amount", "savings", "employment",
           "installment_rate", "personal_status", "debtors", "residence_since", "property", "age",
           "other_plans", "housing", "n_credits", "job", "n_dependents", "telephone",
           "foreign_worker", "target"]

LABELS = {
    "checking": {"A11": "<0", "A12": "0-200", "A13": ">=200", "A14": "none"},
    "history": {"A30": "none/all paid", "A31": "all paid (this bank)", "A32": "existing paid",
                "A33": "past delay", "A34": "critical"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture", "A43": "radio/tv",
                "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
                "A48": "retraining", "A49": "business", "A410": "other"},
    "savings": {"A61": "<100", "A62": "100-500", "A63": "500-1000", "A64": ">=1000", "A65": "unknown/none"},
    "employment": {"A71": "unemployed", "A72": "<1y", "A73": "1-4y", "A74": "4-7y", "A75": ">=7y"},
    "personal_status": {"A91": "male div/sep", "A92": "female div/sep/mar", "A93": "male single",
                        "A94": "male mar/wid", "A95": "female single"},
    "debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings/insurance", "A123": "car/other", "A124": "none"},
    "other_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "free"},
    "job": {"A171": "unskilled non-res", "A172": "unskilled res", "A173": "skilled", "A174": "highly skilled"},
    "telephone": {"A191": "no", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

RARE_PURPOSE = ["appliances", "repairs", "other", "retraining", "vacation"]
EMPLOYMENT_GROUP = {"unemployed": "<1y/unemployed", "<1y": "<1y/unemployed", "1-4y": "1-4y",
                    "4-7y": ">=4y", ">=7y": ">=4y"}

NUM = ["duration", "log_amount", "log_monthly_payment", "installment_rate", "residence_since",
       "age", "n_credits", "n_dependents"]
CAT = ["checking", "history", "purpose_grp", "savings", "employment_grp", "debtors", "property",
       "other_plans", "housing", "job", "telephone"]
PROTECTED = ["sex", "age_group", "foreign_worker"]  # kept out of the model, used for fairness checks


def load_raw(path=DATA):
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMNS)
    for col, mapping in LABELS.items():
        df[col] = df[col].map(mapping)
    df["default"] = (df.pop("target") == 2).astype(int)  # raw coding: 1 = good, 2 = bad
    return df


def engineer(df):
    df = df.copy()
    df["log_amount"] = np.log(df["amount"])
    df["monthly_payment"] = df["amount"] / df["duration"]  # no income field: repayment size is the closest affordability proxy
    df["log_monthly_payment"] = np.log(df["monthly_payment"])
    df["purpose_grp"] = df["purpose"].where(~df["purpose"].isin(RARE_PURPOSE), "other")
    df["employment_grp"] = df["employment"].map(EMPLOYMENT_GROUP)
    df["sex"] = np.where(df["personal_status"].str.startswith("female"), "female", "male")
    df["age_group"] = np.where(df["age"] < 25, "<25", ">=25")
    return df


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ], verbose_feature_names_out=False)


def get_data(test_size=0.2):
    df = engineer(load_raw())
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        df[NUM + CAT], df["default"], df[PROTECTED],
        test_size=test_size, stratify=df["default"], random_state=SEED)
    return {"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
            "groups_train": g_tr, "groups_test": g_te}
