"""Pure, testable feature logic for the Seattle building-energy project.

Keeps the data cleaning and feature preparation free of sklearn/plotting so it
can be unit-tested without the full modelling stack. The heavy pipeline (model
sweep, CV, figures) lives in ``run_models.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Targets the project predicts for non-residential buildings.
TARGET_CO2 = "TotalGHGEmissions"
TARGET_ENERGY = "SiteEnergyUse(kBtu)"

# Numeric predictors available before energy is measured (no leakage).
BASE_FEATURES = [
    "NumberofBuildings",
    "NumberofFloors",
    "PropertyGFATotal",
    "PropertyGFAParking",
    "YearBuilt",
    "LargestPropertyUseTypeGFA",
]


def filter_non_residential(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only non-residential buildings and drop flagged outliers.

    The project scope is non-housing buildings. Rows marked as outliers or with
    a non-compliant status are removed — they bias the regression.
    """
    out = df.copy()
    if "BuildingType" in out.columns:
        mask = ~out["BuildingType"].astype(str).str.contains("Multifamily", case=False, na=False)
        out = out[mask]
    if "Outlier" in out.columns:
        out = out[out["Outlier"].isna() | (out["Outlier"].astype(str).str.strip() == "")]
    return out


def drop_target_leakage(features: list[str]) -> list[str]:
    """Remove any feature that is an energy/emission measurement (leakage).

    Predicting energy from other energy columns (e.g. Electricity, NaturalGas,
    EUI) is leakage — those are only known *after* the fact. This keeps the
    feature list honest.
    """
    banned = ("energyuse", "ghg", "emission", "eui", "electricity", "naturalgas",
              "steam")
    # ENERGYSTARScore is a rating, not a measured energy quantity -> not leakage.
    allow = {"energystarscore"}
    return [
        f for f in features
        if f.lower() in allow or not any(b in f.lower() for b in banned)
    ]


def prepare_xy(df: pd.DataFrame, target: str, features: list[str] | None = None,
               log_target: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Build a clean (X, y) for a given target.

    - selects numeric ``features`` (defaults to BASE_FEATURES, leakage-free),
    - drops rows with a missing/zero/negative target,
    - log1p-transforms the target by default (both targets are heavily
      right-skewed, like most energy/emission quantities).
    """
    features = features or BASE_FEATURES
    features = drop_target_leakage(features)
    data = df.copy()
    data = data[data[target].notna() & (data[target] > 0)]
    X = data[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = data[target].astype(float)
    if log_target:
        y = np.log1p(y)
    return X, y


def target_skew(df: pd.DataFrame, target: str) -> float:
    """Skewness of a target column on positive values (to justify log1p)."""
    s = df[target]
    s = s[s.notna() & (s > 0)].astype(float)
    return float(s.skew())
