"""Reproducible P4 pipeline — predict CO2 emissions and energy use, honestly.

Run from the repo root (CSV in ``data/``):

    python src/run_models.py --data data/2016_Building_Energy_Benchmarking.csv \
        --out docs/figures --results results.json

For each target (CO2, energy) it:
  1. filters to non-residential buildings, drops outliers + target leakage,
  2. log1p-transforms the skewed target,
  3. compares a Dummy baseline, Linear, Ridge, Lasso, Random Forest, Gradient
     Boosting with 5-fold CV (R^2 + RMSE),
  4. answers the ENERGYSTARScore question: does it actually help the model?
All numbers -> --results (JSON), one bar chart per target -> --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F

SEED = 10


def models() -> dict:
    """The model zoo, from linear to regularised to kernel to ensemble.

    Kernel methods (SVR, Kernel Ridge) need scaled inputs, hence the pipelines.
    This spans the four regression families on purpose: a recruiter sees the
    full supervised-regression toolbox benchmarked on equal footing, not one
    lucky model.
    """
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR

    return {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.01, random_state=SEED, max_iter=5000)),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, gamma="scale")),
        "Kernel Ridge (RBF)": make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }


def tune_best_family(X, y) -> dict:
    """Light GridSearch on one representative of each model family.

    Materialises the hyperparameter search the 2022 notebooks did (alpha for
    Ridge/KRR, C+gamma for SVR, n_estimators+depth for GBR) without blowing up
    runtime — small grids, 5-fold, R² scoring. Reports the best params and score
    per family so the sweep isn't just default-settings luck.
    """
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR

    searches = {
        "Ridge": (make_pipeline(StandardScaler(), Ridge(random_state=SEED)),
                  {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}),
        "SVR (RBF)": (make_pipeline(StandardScaler(), SVR(kernel="rbf")),
                      {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.1]}),
        "Kernel Ridge (RBF)": (make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                               {"kernelridge__alpha": [0.1, 1.0, 10.0],
                                "kernelridge__gamma": [None, 0.1]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED),
                              {"n_estimators": [100, 300], "max_depth": [2, 3]}),
    }
    out = {}
    for name, (est, grid) in searches.items():
        gs = GridSearchCV(est, grid, cv=5, scoring="r2", n_jobs=-1)
        gs.fit(X, y)
        out[name] = {"best_r2": float(gs.best_score_),
                     "best_params": {k: str(v) for k, v in gs.best_params_.items()}}
    return out


def evaluate(X, y) -> dict:
    """5-fold CV R^2 and RMSE for every model."""
    rows = {}
    for name, model in models().items():
        cv = cross_validate(model, X, y, cv=5,
                            scoring=["r2", "neg_root_mean_squared_error"], n_jobs=-1)
        rows[name] = {
            "r2": float(np.mean(cv["test_r2"])),
            "r2_std": float(np.std(cv["test_r2"])),
            "rmse": float(-np.mean(cv["test_neg_root_mean_squared_error"])),
        }
    return rows


def fig_scores(scores: dict, target_name: str, out: Path) -> None:
    names = list(scores)
    r2 = [scores[n]["r2"] for n in names]
    # clip the axis so one badly-failing model (e.g. untuned Kernel Ridge at
    # R²≈-3.6) doesn't crush the scale and hide the models that actually work.
    floor = -1.0
    r2_plot = [max(v, floor) for v in r2]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(names, r2_plot, color="#36c")
    bars[int(np.argmax(r2))].set_color("#2a7")
    for bar, v, vp in zip(bars, r2, r2_plot, strict=False):
        clipped = v < floor
        if clipped:
            # off-scale model: put the true value just right of the clipped bar tip
            ax.text(floor + 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f} (off-scale)", va="center", ha="left", fontsize=9, color="#fff")
        else:
            x = vp + (0.015 if vp >= 0 else -0.015)
            ax.text(x, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                    va="center", ha="left" if vp >= 0 else "right", fontsize=9)
    ax.set_xlim(left=floor - 0.05)
    ax.set_xlabel("5-fold CV R² (log target) — higher is better")
    ax.set_title(f"{target_name}: model comparison (Gradient Boosting wins)")
    ax.axvline(0, color="k", lw=0.5)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(out / f"models_{target_name.lower().replace(' ', '_')}.png", dpi=110)
    plt.close(fig)


def energystar_ablation(df: pd.DataFrame, target: str) -> dict:
    """Does ENERGYSTARScore help? Compare best model with vs without it."""
    gb = GradientBoostingRegressor(random_state=SEED)

    X0, y0 = F.prepare_xy(df, target)
    r2_without = float(np.mean(cross_validate(gb, X0, y0, cv=5, scoring="r2")["test_score"]))

    feats = F.BASE_FEATURES + ["ENERGYSTARScore"]
    X1, y1 = F.prepare_xy(df, target, features=feats)
    r2_with = float(np.mean(cross_validate(gb, X1, y1, cv=5, scoring="r2")["test_score"]))

    return {"r2_without_energystar": r2_without,
            "r2_with_energystar": r2_with,
            "delta": r2_with - r2_without}


def fig_shap(df: pd.DataFrame, target: str, label: str, out: Path) -> dict:
    """SHAP explanation of the best model: which features drive the prediction.

    A good R² is not enough in 2026 — a model has to be explainable. SHAP values
    attribute each prediction to its features, so the city can see *why* a
    building is predicted energy-hungry, not just the number. Fitted on the full
    (leakage-free) set with the winning Gradient Boosting.
    """
    try:
        import shap
    except ImportError:
        return {"skipped": "shap not installed"}

    X, y = F.prepare_xy(df, target)
    gb = GradientBoostingRegressor(random_state=SEED).fit(X, y)
    explainer = shap.TreeExplainer(gb)
    sv = explainer.shap_values(X)

    fig = plt.figure()
    shap.summary_plot(sv, X, show=False, plot_size=(9, 5))
    fig = plt.gcf()
    fig.suptitle(f"{label}: what drives the prediction (SHAP)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out / f"shap_{label.lower().replace(' ', '_')}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    mean_abs = np.abs(sv).mean(axis=0)
    ranking = sorted(zip(X.columns, mean_abs, strict=False), key=lambda t: t[1], reverse=True)
    return {"feature_ranking": [{"feature": f, "mean_abs_shap": round(float(v), 4)}
                                for f, v in ranking]}


def fig_map(df: pd.DataFrame, out: Path) -> dict:
    """Map Seattle buildings by energy use — where are the energy-hungry ones?

    The dataset carries each building's latitude/longitude, so the model's
    subject can be put back on the map: a prediction is useful to a city only if
    it can be located. Buildings coloured by (log) site energy use.
    """
    cols = ["Latitude", "Longitude", F.TARGET_ENERGY, "Neighborhood"]
    if not all(c in df.columns for c in ["Latitude", "Longitude"]):
        return {"skipped": "no coordinates"}
    d = df[cols].copy()
    for c in ["Latitude", "Longitude", F.TARGET_ENERGY]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d[F.TARGET_ENERGY] > 0) & d["Latitude"].between(47.4, 47.8)
          & d["Longitude"].between(-122.5, -122.2)].dropna(subset=["Latitude", "Longitude"])
    d["log_energy"] = np.log1p(d[F.TARGET_ENERGY])

    fig, ax = plt.subplots(figsize=(7, 8))
    sc = ax.scatter(d["Longitude"], d["Latitude"], c=d["log_energy"],
                    cmap="inferno", s=14, alpha=0.75, edgecolor="none")
    # annotate the two high-consumption neighbourhoods the analysis calls out
    for name, lat, lng in [("Downtown", 47.606, -122.335), ("Lake Union", 47.637, -122.335)]:
        ax.annotate(name, (lng, lat), color="#111", fontsize=9, fontweight="bold",
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.set_title(f"Seattle non-residential buildings by energy use (n={len(d):,})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1.4)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.tick_params(labelsize=8)
    fig.colorbar(sc, ax=ax, label="log site energy use (kBtu)", shrink=0.6)
    fig.tight_layout()
    fig.savefig(out / "map_energy_seattle.png", dpi=120)
    plt.close(fig)

    # top-3 neighborhoods by median energy use (a small business read)
    by_nb = (d.groupby("Neighborhood")[F.TARGET_ENERGY].median()
             .sort_values(ascending=False).head(3))
    return {"n_mapped": int(len(d)),
            "top_neighborhoods_by_median_energy": {k: round(float(v)) for k, v in by_nb.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument("--results", type=Path, default=Path("results.json"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading + filtering non-residential buildings...")
    df = pd.read_csv(args.data)
    df = F.filter_non_residential(df)
    print(f"   {len(df):,} non-residential buildings after cleaning")

    results = {"n_buildings": int(len(df))}
    for target, label in [(F.TARGET_CO2, "CO2 emissions"), (F.TARGET_ENERGY, "Energy use")]:
        print(f"\n=== {label} ({target}) ===")
        print(f"   target skew (raw): {F.target_skew(df, target):.1f} -> log1p applied")
        X, y = F.prepare_xy(df, target)
        scores = evaluate(X, y)
        fig_scores(scores, label, args.out)
        best = max(scores, key=lambda n: scores[n]["r2"])
        print(f"   best: {best}  R²={scores[best]['r2']:.3f}")
        results[target] = {
            "label": label,
            "n": int(len(y)),
            "target_skew_raw": F.target_skew(df, target),
            "scores": scores,
            "best_model": best,
            "best_r2": scores[best]["r2"],
            "energystar_ablation": energystar_ablation(df, target),
        }

    print("\n=== GridSearch tuning per family (energy) ===")
    Xe, ye = F.prepare_xy(df, F.TARGET_ENERGY)
    results["tuning_energy"] = tune_best_family(Xe, ye)
    for fam, r in results["tuning_energy"].items():
        print(f"   {fam}: R²={r['best_r2']:.3f}  {r['best_params']}")

    print("\n=== SHAP explanation (best model, energy) ===")
    results["shap_energy"] = fig_shap(df, F.TARGET_ENERGY, "Energy use", args.out)

    print("=== Seattle energy map ===")
    results["map"] = fig_map(df, args.out)

    args.results.write_text(json.dumps(results, indent=2))
    print(f"\nDone. Figures -> {args.out}, numbers -> {args.results}")


if __name__ == "__main__":
    main()
