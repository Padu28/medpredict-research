"""Leakage-safe splitting: temporal + patient-level, with explicit checks."""
from __future__ import annotations

import pandas as pd


def temporal_patient_split(df: pd.DataFrame, id_col: str, time_col: str, train_end: str, validation_end: str):
    """Assign each PATIENT to the split of their first discharge, so no patient spans splits.

    Later discharges of a patient follow their first one; test therefore contains only
    patients first seen after ``validation_end`` (strict temporal validation).
    """
    first = df.groupby(id_col)[time_col].min()
    split = pd.Series("test", index=first.index)
    split[first <= pd.Timestamp(train_end)] = "train"
    split[(first > pd.Timestamp(train_end)) & (first <= pd.Timestamp(validation_end))] = "validation"
    lab = df[id_col].map(split)
    return df[lab == "train"].copy(), df[lab == "validation"].copy(), df[lab == "test"].copy()


def check_no_patient_overlap(*frames: pd.DataFrame, id_col: str = "patient_id") -> bool:
    seen: set = set()
    for f in frames:
        ids = set(f[id_col])
        if seen & ids:
            return False
        seen |= ids
    return True


def check_no_leakage(features: list[str], leakage_columns: list[str], target: str) -> list[str]:
    """Return offending columns (empty list = OK)."""
    return [c for c in features if c in set(leakage_columns) | {target}]
