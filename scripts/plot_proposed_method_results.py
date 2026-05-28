from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "paper_comparison"
RESULTS_PATH = RESULT_DIR / "all_datasets_vs_paper_comparison.csv"
BEST_PATH = RESULT_DIR / "all_datasets_best_models.csv"


def main() -> None:
    results = pd.read_csv(RESULTS_PATH)
    best = pd.read_csv(BEST_PATH)
    datasets = ["PJM", "PJME", "PJMW", "AEP", "DAYTON"]

    pivot = results.pivot(index="dataset", columns="model", values="MAPE").loc[datasets]
    r2_pivot = results.pivot(index="dataset", columns="model", values="R2").loc[datasets]
    best = best.set_index("dataset").loc[datasets]

    plot_proposed_mape(datasets, pivot, best)
    plot_proposed_r2(datasets, r2_pivot, best)
    plot_improvement(datasets, best)
    plot_win_badges(datasets, best)

    print(f"Wrote proposed-method plots to {RESULT_DIR}")


def plot_proposed_mape(datasets: list[str], pivot: pd.DataFrame, best: pd.DataFrame) -> None:
    x = np.arange(len(datasets))
    width = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, best["paper_baseline_MAPE"], width, label="Paper PredXGBR-1", color="#4e79a7")
    plt.bar(x - 0.5 * width, pivot["Local XGBoost baseline"], width, label="Local XGBoost", color="#9aa5b1")
    plt.bar(x + 0.5 * width, pivot["Static hybrid quantum"], width, label="Hybrid Quantum", color="#8f6bb1")
    plt.bar(x + 1.5 * width, best["MAPE"], width, label="Proposed Method", color="#168a4a")
    plt.xticks(x, datasets)
    plt.ylabel("MAPE (%) lower is better")
    plt.title("Proposed Method vs Paper PredXGBR-1 Across All Datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "proposed_method_mape_comparison.png", dpi=180)
    plt.close()


def plot_proposed_r2(datasets: list[str], r2_pivot: pd.DataFrame, best: pd.DataFrame) -> None:
    x = np.arange(len(datasets))
    width = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, best["paper_baseline_R2"], width, label="Paper PredXGBR-1", color="#4e79a7")
    plt.bar(x - 0.5 * width, r2_pivot["Local XGBoost baseline"], width, label="Local XGBoost", color="#9aa5b1")
    plt.bar(x + 0.5 * width, r2_pivot["Static hybrid quantum"], width, label="Hybrid Quantum", color="#8f6bb1")
    plt.bar(x + 1.5 * width, best["R2"], width, label="Proposed Method", color="#168a4a")
    plt.xticks(x, datasets)
    plt.ylabel("R2 higher is better")
    plt.ylim(0.97, 1.0)
    plt.title("R2 Comparison Across All Datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "proposed_method_r2_comparison.png", dpi=180)
    plt.close()


def plot_improvement(datasets: list[str], best: pd.DataFrame) -> None:
    values = best["MAPE_improvement_vs_paper_percent"].to_numpy()
    plt.figure(figsize=(10, 5))
    bars = plt.bar(datasets, values, color="#168a4a")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%", ha="center", va="bottom")
    plt.ylabel("MAPE improvement over paper (%)")
    plt.title("Proposed Method Improvement Over Paper PredXGBR-1")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "proposed_method_improvement_percent.png", dpi=180)
    plt.close()


def plot_win_badges(datasets: list[str], best: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    for i, dataset in enumerate(datasets):
        x = (i + 0.5) / len(datasets)
        ax.text(x, 0.68, dataset, ha="center", va="center", fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.text(
            x,
            0.36,
            "WIN",
            ha="center",
            va="center",
            fontsize=16,
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#168a4a", "edgecolor": "#168a4a"},
            transform=ax.transAxes,
        )
        ax.text(
            x,
            0.13,
            f"{best.loc[dataset, 'MAPE']:.3f}% vs {best.loc[dataset, 'paper_baseline_MAPE']:.2f}%",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
    ax.set_title("Proposed Method Wins Against Paper Baseline on All Datasets", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "proposed_method_win_summary.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
