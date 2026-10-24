"""Unit tests for the pure feature logic — no CSV, no sklearn needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as F  # noqa: E402


def test_drop_target_leakage_removes_energy_columns():
    feats = ["PropertyGFATotal", "Electricity(kBtu)", "SiteEnergyUse(kBtu)",
             "NumberofFloors", "TotalGHGEmissions"]
    kept = F.drop_target_leakage(feats)
    assert "PropertyGFATotal" in kept
    assert "NumberofFloors" in kept
    assert "Electricity(kBtu)" not in kept
    assert "SiteEnergyUse(kBtu)" not in kept
    assert "TotalGHGEmissions" not in kept


def test_filter_non_residential_drops_multifamily():
    df = pd.DataFrame({
        "BuildingType": ["NonResidential", "Multifamily LR (1-4)", "Campus"],
        "Outlier": [np.nan, np.nan, np.nan],
        "TotalGHGEmissions": [10, 20, 30],
    })
    out = F.filter_non_residential(df)
    assert len(out) == 2
    assert not out["BuildingType"].str.contains("Multifamily").any()


def test_prepare_xy_drops_nonpositive_target_and_logs():
    df = pd.DataFrame({
        "NumberofBuildings": [1, 1, 1],
        "NumberofFloors": [2, 3, 4],
        "PropertyGFATotal": [1000, 2000, 3000],
        "PropertyGFAParking": [0, 0, 0],
        "YearBuilt": [1990, 2000, 2010],
        "LargestPropertyUseTypeGFA": [900, 1800, 2700],
        "TotalGHGEmissions": [100.0, 0.0, 50.0],  # the 0 must be dropped
    })
    X, y = F.prepare_xy(df, "TotalGHGEmissions")
    assert len(y) == 2                       # zero-target row removed
    assert np.isclose(y.iloc[0], np.log1p(100.0))   # log1p applied


def test_prepare_xy_fills_missing_features():
    df = pd.DataFrame({
        "NumberofBuildings": [1, 1],
        "NumberofFloors": [np.nan, 4],
        "PropertyGFATotal": [1000, 2000],
        "PropertyGFAParking": [0, 0],
        "YearBuilt": [1990, 2000],
        "LargestPropertyUseTypeGFA": [900, 1800],
        "TotalGHGEmissions": [100.0, 50.0],
    })
    X, _ = F.prepare_xy(df, "TotalGHGEmissions")
    assert not X.isna().any().any()          # NaN filled with median
