from sklearn.model_selection import StratifiedKFold

SEED = 42
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

EMPLOYMENT_LEVEL = {"unemployed": 0, "<1y": 1, "1-4y": 2, "4-7y": 3, ">=7y": 4}
PROPERTY_LEVEL = {"none": 0, "car/other": 1, "savings/insurance": 2, "real estate": 3}
JOB_LEVEL = {"unskilled non-res": 0, "unskilled res": 1, "skilled": 2, "highly skilled": 3}
HOUSING_POINTS = {"free": 0, "rent": 1, "own": 2}
PURPOSE_TYPE = {"education": "professional", "retraining": "professional", "business": "professional",
                "new car": "needs", "used car": "needs", "furniture": "needs", "repairs": "needs",
                "vacation": "wants", "radio/tv": "wants",
                "appliances": "others", "other": "others"}

FEATURES = ["duration", "log_amount", "log_monthly_payment", "age", "employment_level", "property_level",
            "installment_rate", "n_credits", "checking", "savings", "history", "purpose_type",
            "other_plans", "housing", "has_guarantor", "has_coapplicant", "thin_buffer",
            "duration_x_rate", "stability_score", "asset_score", "job_level"]

PROTECTED = ["sex", "foreign_worker"]  # fairness only, never model inputs

NUMERIC = ["duration", "log_amount", "log_monthly_payment", "age", "duration_x_rate", "stability_score"]
CATEGORICAL = ["checking", "savings", "history", "purpose_type", "other_plans", "housing"]
# everything else in FEATURES (ordinal and binary) passes through unchanged