"""Portfolio-styled figures (dark AnastatsIA charter) for the P4 website cards.

Separate from the reproducible pipeline: the *pretty* versions (dark background,
azure accent) for the portfolio gallery, not the analytical outputs.

    python docs/make_portfolio_figures.py --data data/2016_Building_Energy_Benchmarking.csv \
        --out ../../../../05_PROJETS/04_PORTFOLIO/public/projects
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as F  # noqa: E402

BG = "#0b1424"
PANEL = "#0e1b2e"
AZURE = "#38bdf8"
INK = "#eaf4ff"
MUTE = "#9fc7e8"


def _style(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#1e3a5f")
    ax.tick_params(colors=MUTE, labelsize=8)
    ax.yaxis.label.set_color(MUTE)
    ax.xaxis.label.set_color(MUTE)
    ax.title.set_color(INK)


def make_map(df: pd.DataFrame, out: Path):
    d = df[["Latitude", "Longitude", F.TARGET_ENERGY, "Neighborhood"]].copy()
    for c in ["Latitude", "Longitude", F.TARGET_ENERGY]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d[F.TARGET_ENERGY] > 0) & d["Latitude"].between(47.4, 47.8)
          & d["Longitude"].between(-122.5, -122.2)].dropna(subset=["Latitude", "Longitude"])
    d["log_energy"] = np.log1p(d[F.TARGET_ENERGY])

    fig, ax = plt.subplots(figsize=(7, 8))
    _style(ax, fig)
    sc = ax.scatter(d["Longitude"], d["Latitude"], c=d["log_energy"],
                    cmap="inferno", s=15, alpha=0.8, edgecolor="none")
    for name, lat, lng in [("Downtown", 47.606, -122.335), ("Lake Union", 47.637, -122.335)]:
        ax.annotate(name, (lng, lat), color=INK, fontsize=10, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#0b1424", ec=AZURE, alpha=0.8))
    ax.set_title("Seattle buildings by energy use\n— the Downtown / Lake Union corridor", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1.4)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    cb = fig.colorbar(sc, ax=ax, label="log site energy use (kBtu)", shrink=0.55)
    cb.ax.yaxis.label.set_color(MUTE)
    cb.ax.tick_params(colors=MUTE)
    fig.tight_layout()
    fig.savefig(out / "energy-prediction-map.png", dpi=125, facecolor=BG)
    plt.close(fig)


def make_shap_bar(df: pd.DataFrame, out: Path):
    """Dark-charter SHAP importance as a clean bar (mean |SHAP|)."""
    try:
        import shap
    except ImportError:
        return
    X, y = F.prepare_xy(df, F.TARGET_ENERGY)
    gb = GradientBoostingRegressor(random_state=10).fit(X, y)
    sv = shap.TreeExplainer(gb).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style(ax, fig)
    ax.barh(imp.index, imp.values, color=AZURE)
    for i, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, i, f"{v:.2f}", va="center", color=MUTE, fontsize=9)
    ax.set_xlabel("mean |SHAP| — impact on predicted energy use")
    ax.set_title("What drives the prediction: building size dominates", fontsize=12)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(out / "energy-prediction-shap.png", dpi=125, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = F.filter_non_residential(pd.read_csv(args.data))
    make_map(df, args.out)
    make_shap_bar(df, args.out)
    print(f"portfolio figures -> {args.out}")


if __name__ == "__main__":
    main()
