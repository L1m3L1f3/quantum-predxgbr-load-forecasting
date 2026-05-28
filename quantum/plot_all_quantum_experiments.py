from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from baseline_pipeline import RESULTS_DIR


OUTPUT_DIR = RESULTS_DIR / "quantum_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare all classical, quantum, and hybrid experiment outputs.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def build_all_experiment_comparison(args: argparse.Namespace) -> pd.DataFrame:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(load_full_xgboost())
    rows.extend(load_quantum_feature())
    rows.extend(load_hybrid_residual())
    rows.extend(load_quantum_augmented())
    rows.extend(load_matched_sweep())
    rows.extend(load_postprocess_sweep())
    rows.extend(load_hybrid_quantum_postprocess())

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(["comparison_scope", "MAE", "model"])
    comparison.to_csv(args.output_dir / "all_quantum_experiment_comparison.csv", index=False)
    plot_all_experiments(comparison, args.output_dir / "all_quantum_experiment_comparison.png")
    plot_scoped_best(comparison, args.output_dir / "all_quantum_scoped_best.png")

    print(comparison.to_string(index=False))
    print(f"Saved comparison outputs to {args.output_dir}")
    return comparison


def load_full_xgboost() -> list[dict]:
    path = RESULTS_DIR / "metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        normalize_row(row, "full_test_strong_baseline", "xgboost_full_baseline", "full test, all training rows")
        for row in df.to_dict("records")
    ]


def load_quantum_feature() -> list[dict]:
    path = RESULTS_DIR / "quantum" / "quantum_metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        normalize_row(row, "sampled_quantum_only", row["model"], "sampled train/test rows")
        for row in df.to_dict("records")
    ]


def load_hybrid_residual() -> list[dict]:
    path = RESULTS_DIR / "hybrid_quantum_residual" / "hybrid_quantum_residual_metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        normalize_row(row, "full_test_residual", row["model"], "full test, residual correction")
        for row in df.to_dict("records")
    ]


def load_quantum_augmented() -> list[dict]:
    path = RESULTS_DIR / "quantum_augmented_xgboost" / "quantum_augmented_xgboost_metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        normalize_row(row, "sampled_augmented_xgboost", row["model"], "same sampled rows")
        for row in df.to_dict("records")
    ]


def load_matched_sweep() -> list[dict]:
    path = RESULTS_DIR / "quantum_matched_sweep" / "quantum_matched_model_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for row in df.to_dict("records"):
        rows.append(
            {
                "dataset": "PJME",
                "comparison_scope": "matched_low_data_sweep_mean",
                "model": row["model"],
                "MAE": row["mean_MAE"],
                "RMSE": row["mean_RMSE"],
                "MAPE": row["mean_MAPE"],
                "R2": row["mean_R2"],
                "training_time_seconds": None,
                "notes": f"mean across {int(row['runs'])} matched sweep runs",
            }
        )
    return rows


def load_postprocess_sweep() -> list[dict]:
    path = RESULTS_DIR / "postprocess_baseline_sweep" / "postprocess_baseline_sweep_metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        normalize_row(row, "full_test_postprocess_sweep", row["model"], row["notes"])
        for row in df.to_dict("records")
    ]


def load_hybrid_quantum_postprocess() -> list[dict]:
    path = RESULTS_DIR / "hybrid_quantum_postprocess" / "hybrid_quantum_postprocess_metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        normalize_row(row, "full_test_hybrid_quantum_postprocess", row["model"], row["notes"])
        for row in df.to_dict("records")
    ]


def normalize_row(row: dict, scope: str, model: str, notes: str) -> dict:
    return {
        "dataset": row.get("dataset", "PJME"),
        "comparison_scope": scope,
        "model": model,
        "MAE": row["MAE"],
        "RMSE": row["RMSE"],
        "MAPE": row["MAPE"],
        "R2": row["R2"],
        "training_time_seconds": row.get("training_time_seconds"),
        "notes": notes,
    }


def plot_all_experiments(comparison: pd.DataFrame, output_path: Path) -> None:
    ordered = comparison.sort_values("MAE", ascending=True)
    labels = ordered.apply(lambda row: f"{row['comparison_scope']}: {row['model']}", axis=1)
    plt.figure(figsize=(14, max(7, len(ordered) * 0.35)))
    plt.barh(labels[::-1], ordered["MAE"].to_numpy()[::-1], color=[scope_color(scope) for scope in ordered["comparison_scope"][::-1]])
    plt.xlabel("MAE lower is better")
    plt.title("All Available Classical, Quantum, and Hybrid Results")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_scoped_best(comparison: pd.DataFrame, output_path: Path) -> None:
    best = comparison.sort_values(["comparison_scope", "MAE"]).groupby("comparison_scope", as_index=False).first()
    plt.figure(figsize=(11, 5))
    plt.barh(best["comparison_scope"], best["MAE"], color=[scope_color(scope) for scope in best["comparison_scope"]])
    for index, row in best.iterrows():
        plt.text(row["MAE"], index, f"  {row['model']}", va="center", fontsize=8)
    plt.xlabel("Best MAE within each comparison scope")
    plt.title("Best Result Per Scope")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def scope_color(scope: str) -> str:
    if "strong_baseline" in scope:
        return "#4e79a7"
    if "matched" in scope:
        return "#2a9d8f"
    if "residual" in scope:
        return "#f28e2b"
    if "hybrid_quantum_postprocess" in scope:
        return "#8f6bb1"
    if "postprocess" in scope:
        return "#2a9d8f"
    if "augmented" in scope:
        return "#59a14f"
    return "#e15759"


if __name__ == "__main__":
    build_all_experiment_comparison(parse_args())
