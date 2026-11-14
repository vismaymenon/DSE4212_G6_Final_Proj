from pathlib import Path
import pandas as pd
from constants import (COLUMNS,LABELS)


DATA = Path(__file__).resolve().parent / "data" / "german.data"

def load_data(path=DATA):
    """
    Read german.data and return the full dataframe.
    Change the label (default) to 0/1 instead of 1/2
    """
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMNS)
    for col, mapping in LABELS.items():
        df[col] = df[col].map(mapping)
    df["default"] = (df.pop("target") == 2).astype(int)  # raw coding: 1 = good, 2 = bad
    return df