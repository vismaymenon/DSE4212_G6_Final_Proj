import numpy as np
from sklearn.model_selection import train_test_split
from constants import (SEED, EMPLOYMENT_LEVEL,PROPERTY_LEVEL,JOB_LEVEL,HOUSING_POINTS,PURPOSE_TYPE,
                       FEATURES,PROTECTED)
from load_data import load_data


def engineer(df):
    """Stateless features only: every row is transformed independently of the others."""
    df = df.copy()
    df["log_amount"] = np.log(df["amount"])
    df["log_monthly_payment"] = np.log(df["amount"] / df["duration"])  # no income field: affordability proxy
    df["employment_level"] = df["employment"].map(EMPLOYMENT_LEVEL)
    df["property_level"] = df["property"].map(PROPERTY_LEVEL)
    df["job_level"] = df["job"].map(JOB_LEVEL)
    df["purpose_type"] = df["purpose"].map(PURPOSE_TYPE)
    df["has_guarantor"] = (df["debtors"] == "guarantor").astype(int)
    df["has_coapplicant"] = (df["debtors"] == "co-applicant").astype(int)
    df["thin_buffer"] = ((df["checking"] == "<0") & (df["savings"] == "<100")).astype(int)
    df["duration_x_rate"] = df["duration"] * df["installment_rate"]
    # each component scaled to 0 to 1 so none dominates the sum
    df["stability_score"] = (df["employment_level"] / 4
                             + (df["residence_since"] - 1) / 3
                             + df["housing"].map(HOUSING_POINTS) / 2)
    df["asset_score"] = ((df["property"] == "real estate").astype(int)
                         + (df["housing"] == "own").astype(int)
                         + df["savings"].isin(["500-1000", ">=1000"]).astype(int))
    df["sex"] = np.where(df["personal_status"].str.startswith("female"), "female", "male")
    return df


def get_data(test_size=0.2):
    df = engineer(load_data())
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=df["default"], random_state=SEED)
    train, test = df.loc[train_idx].copy(), df.loc[test_idx].copy()

    return {"X_train": train[FEATURES], "X_test": test[FEATURES],
            "y_train": train["default"], "y_test": test["default"],
            "groups_train": train[PROTECTED], "groups_test": test[PROTECTED]}