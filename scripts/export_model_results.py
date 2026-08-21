from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PredXGBR run outputs to a downloadable Excel workbook and visualization graphs."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--prefix", default="predxgbr_local_dataset_results")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}. Run python run_all.py first.")
    return path


def load_config(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text())
    return pd.DataFrame([data])


def plot_residuals(predictions: pd.DataFrame, output_path: Path) -> None:
    residuals = predictions["actual_load"] - predictions["predicted_load"]
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, color="#2f80ed", edgecolor="white")
    plt.title("Prediction Residual Distribution")
    plt.xlabel("Residual (actual - predicted load MW)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_over_time(predictions: pd.DataFrame, output_path: Path) -> None:
    predictions = predictions.copy()
    predictions["datetime"] = pd.to_datetime(predictions["datetime"])
    predictions["absolute_error"] = (predictions["actual_load"] - predictions["predicted_load"]).abs()
    daily_error = predictions.set_index("datetime")["absolute_error"].resample("D").mean()

    plt.figure(figsize=(14, 5))
    plt.plot(daily_error.index, daily_error.values, linewidth=1, color="#eb5757")
    plt.title("Daily Mean Absolute Forecast Error")
    plt.xlabel("Date")
    plt.ylabel("Mean absolute error (MW)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def export_results(results_dir: Path, output_dir: Path, prefix: str) -> tuple[Path, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(require_file(results_dir / "metrics.csv"))
    predictions = pd.read_csv(require_file(results_dir / "predictions.csv"))
    history = pd.read_csv(require_file(results_dir / "training_history.csv"))
    config = load_config(results_dir / "run_config.json")

    residuals_path = output_dir / f"{prefix}_residual_distribution.png"
    error_path = output_dir / f"{prefix}_daily_absolute_error.png"
    plot_residuals(predictions, residuals_path)
    plot_error_over_time(predictions, error_path)

    workbook_path = output_dir / f"{prefix}.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        config.to_excel(writer, sheet_name="run_config", index=False)
        predictions.to_excel(writer, sheet_name="predictions", index=False)
        history.to_excel(writer, sheet_name="training_history", index=False)

    return workbook_path, [residuals_path, error_path]


def main() -> None:
    args = parse_args()
    workbook, graphs = export_results(args.results_dir, args.output_dir, args.prefix)
    print(f"Excel report: {workbook}")
    for graph in graphs:
        print(f"Graph: {graph}")


if __name__ == "__main__":
    main()
