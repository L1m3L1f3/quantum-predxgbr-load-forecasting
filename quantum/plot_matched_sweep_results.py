from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline_pipeline import RESULTS_DIR
from quantum.quantum_matched_sweep import OUTPUT_DIR, summarize_winners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and summarize a completed matched quantum sweep.")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=OUTPUT_DIR / "quantum_matched_sweep_results.csv",
        help="CSV created by run_quantum_matched_sweep.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def analyze_completed_sweep(args: argparse.Namespace) -> None:
    results = pd.read_csv(args.results_file)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    winners = summarize_winners(results)
    summary = summarize_by_model(results, winners)
    summary.to_csv(output_dir / "quantum_matched_model_summary.csv", index=False)
    winners.to_csv(output_dir / "quantum_matched_sweep_winners.csv", index=False)

    plot_model_mean_mae(summary, output_dir / "quantum_matched_model_mean_mae.png")
    plot_model_mae_distribution(results, output_dir / "quantum_matched_model_mae_distribution.png")
    plot_winner_counts(winners, output_dir / "quantum_matched_winner_counts.png")
    plot_improvements(winners, output_dir / "quantum_matched_improvements.png")
    plot_setting_grid(results, output_dir / "quantum_matched_setting_grid.png")

    print("Model summary:")
    print(summary.to_string(index=False))
    print(f"\nQuantum/hybrid wins: {int(winners['beats_matched_classical'].sum())} of {len(winners)} settings")
    print(f"Saved plots to {output_dir}")


def summarize_by_model(results: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("model")
        .agg(
            runs=("MAE", "size"),
            mean_MAE=("MAE", "mean"),
            median_MAE=("MAE", "median"),
            best_MAE=("MAE", "min"),
            mean_RMSE=("RMSE", "mean"),
            mean_MAPE=("MAPE", "mean"),
            mean_R2=("R2", "mean"),
        )
        .reset_index()
    )
    winner_counts = winners.groupby("model").size().rename("overall_wins").reset_index()
    matched_win_counts = (
        winners[winners["beats_matched_classical"]].groupby("model").size().rename("matched_classical_wins").reset_index()
    )
    summary = summary.merge(winner_counts, on="model", how="left")
    summary = summary.merge(matched_win_counts, on="model", how="left")
    summary[["overall_wins", "matched_classical_wins"]] = summary[["overall_wins", "matched_classical_wins"]].fillna(0).astype(int)
    return summary.sort_values(["mean_MAE", "best_MAE"])


def plot_model_mean_mae(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values("mean_MAE", ascending=True)
    plt.figure(figsize=(11, 6))
    plt.barh(ordered["model"], ordered["mean_MAE"], color=model_colors(ordered["model"]))
    plt.xlabel("Mean MAE across all sweep runs")
    plt.title("Average Error by Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_mae_distribution(results: pd.DataFrame, output_path: Path) -> None:
    ordered_models = results.groupby("model")["MAE"].median().sort_values().index.tolist()
    data = [results.loc[results["model"] == model, "MAE"].to_numpy() for model in ordered_models]
    plt.figure(figsize=(12, 6))
    box = plt.boxplot(data, labels=ordered_models, patch_artist=True, showfliers=True)
    for patch, model in zip(box["boxes"], ordered_models):
        patch.set_facecolor(model_color(model))
    plt.ylabel("MAE")
    plt.title("MAE Distribution Across All Seeds, Qubits, and Layers")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_winner_counts(winners: pd.DataFrame, output_path: Path) -> None:
    counts = winners["model"].value_counts().sort_values(ascending=True)
    plt.figure(figsize=(10, 5))
    plt.barh(counts.index, counts.values, color=model_colors(counts.index))
    plt.xlabel("Number of settings won")
    plt.title("Best Model Count Across Matched Settings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_improvements(winners: pd.DataFrame, output_path: Path) -> None:
    ordered = winners.sort_values("MAE_improvement_vs_best_classical", ascending=True).copy()
    labels = ordered.apply(lambda row: f"s{row['seed']} q{row['n_qubits']} l{row['n_layers']}", axis=1)
    colors = np.where(ordered["beats_matched_classical"], "#2a9d8f", "#777777")
    plt.figure(figsize=(11, 6))
    plt.barh(labels, ordered["MAE_improvement_vs_best_classical"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("MAE improvement over best matched classical model")
    plt.title("Quantum/Hybrid Advantage by Setting")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_setting_grid(results: pd.DataFrame, output_path: Path) -> None:
    settings = (
        results[["seed", "n_qubits", "n_layers"]]
        .drop_duplicates()
        .sort_values(["seed", "n_qubits", "n_layers"])
        .reset_index(drop=True)
    )
    models = results.groupby("model")["MAE"].median().sort_values().index.tolist()
    matrix = np.full((len(models), len(settings)), np.nan)
    for col, setting in settings.iterrows():
        mask = (
            (results["seed"] == setting["seed"])
            & (results["n_qubits"] == setting["n_qubits"])
            & (results["n_layers"] == setting["n_layers"])
        )
        values = results.loc[mask].set_index("model")["MAE"]
        for row, model in enumerate(models):
            matrix[row, col] = values.loc[model]

    plt.figure(figsize=(14, 6))
    image = plt.imshow(matrix, aspect="auto", cmap="viridis_r")
    plt.colorbar(image, label="MAE lower is better")
    plt.yticks(range(len(models)), models)
    plt.xticks(
        range(len(settings)),
        [f"s{row.seed} q{row.n_qubits} l{row.n_layers}" for row in settings.itertuples()],
        rotation=45,
        ha="right",
    )
    plt.title("All Model MAE by Sweep Setting")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def model_color(model: str) -> str:
    if model.startswith("classical"):
        return "#4e79a7"
    if model.startswith("quantum_feature"):
        return "#e15759"
    return "#2a9d8f"


def model_colors(models: pd.Series | pd.Index) -> list[str]:
    return [model_color(str(model)) for model in models]


if __name__ == "__main__":
    analyze_completed_sweep(parse_args())
