from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_pipeline import DATASETS, DATA_DIR, RESULTS_DIR, load_dataset, resolve_data_file, split_features
from scripts.compare_results_to_predxgbr_paper import PAPER_TEXT_PATH, parse_paper_table


OUTPUT_DIR = RESULTS_DIR / "all_paper_dataset_runs"


@dataclass
class DatasetRunConfig:
    feature_mode: str
    n_estimators: int
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style XGBoost baseline on every available paper dataset.")
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="original")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paper = parse_paper_table(PAPER_TEXT_PATH)

    availability_rows = []
    metric_rows = []
    comparison_rows = []

    for dataset in sorted(DATASETS):
        data_path = resolve_data_file(dataset, None)
        available = data_path.exists()
        availability_rows.append(
            {
                "dataset": dataset,
                "expected_file": str(data_path.relative_to(DATA_DIR.parent)),
                "available": available,
                "status": "FOUND" if available else "MISSING",
            }
        )
        if not available:
            print(f"[MISSING] {dataset}: expected {data_path}")
            continue

        print(f"[RUNNING] {dataset}: {data_path}")
        metrics = run_dataset(dataset, args)
        metric_rows.append(metrics)
        comparison_rows.append(compare_to_paper(metrics, paper))

    availability = pd.DataFrame(availability_rows)
    metrics_df = pd.DataFrame(metric_rows)
    comparison = pd.DataFrame(comparison_rows)

    availability.to_csv(OUTPUT_DIR / "dataset_availability.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "all_dataset_baseline_metrics.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "all_dataset_vs_paper_predxgbr.csv", index=False)
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(asdict(DatasetRunConfig(args.feature_mode, args.n_estimators, args.random_state)), indent=2) + "\n")
    write_text_report(availability, comparison)
    write_html_report(availability, comparison)
    plot_comparison(comparison)
    print(f"Wrote all-dataset paper comparison to {OUTPUT_DIR}")


def run_dataset(dataset: str, args: argparse.Namespace) -> dict:
    split_date = DATASETS[dataset]["split_date"]
    df, _, _, data_path = load_dataset(dataset)
    X_train, y_train, X_test, y_test, _, _ = split_features(df, split_date, args.feature_mode)
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=args.random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    seconds = time.perf_counter() - start
    pred = model.predict(X_test)
    return {
        "dataset": dataset,
        "data_file": str(data_path),
        "feature_mode": args.feature_mode,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAPE": np.mean(np.abs((y_test.to_numpy() - pred) / y_test.to_numpy())) * 100,
        "R2": r2_score(y_test, pred),
        "training_time_seconds": seconds,
    }


def compare_to_paper(metrics: dict, paper: pd.DataFrame) -> dict:
    dataset = metrics["dataset"]
    paper_row = paper[(paper["dataset"] == dataset) & (paper["paper_model"] == "PredXGBR")]
    if paper_row.empty:
        return {
            **metrics,
            "paper_MAPE": np.nan,
            "paper_R2": np.nan,
            "beats_paper_MAPE": False,
            "beats_or_ties_paper_R2": False,
            "status": "NO_PAPER_ROW",
        }
    paper_row = paper_row.iloc[0]
    return {
        **metrics,
        "paper_MAPE": float(paper_row["paper_model1_mape"]),
        "paper_R2": float(paper_row["paper_model1_r2"]),
        "MAPE_delta_vs_paper": float(metrics["MAPE"]) - float(paper_row["paper_model1_mape"]),
        "R2_delta_vs_paper": float(metrics["R2"]) - float(paper_row["paper_model1_r2"]),
        "beats_paper_MAPE": float(metrics["MAPE"]) < float(paper_row["paper_model1_mape"]),
        "beats_or_ties_paper_R2": float(metrics["R2"]) >= float(paper_row["paper_model1_r2"]),
        "status": "WIN" if float(metrics["MAPE"]) < float(paper_row["paper_model1_mape"]) and float(metrics["R2"]) >= float(paper_row["paper_model1_r2"]) else "CHECK",
    }


def write_text_report(availability: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "All Paper Dataset Run Summary",
        "=============================",
        "",
        "Dataset availability:",
    ]
    for row in availability.itertuples(index=False):
        marker = "[✓]" if row.available else "[✗]"
        lines.append(f"{marker} {row.dataset}: {row.status} ({row.expected_file})")
    lines.extend(["", "Available dataset results vs paper PredXGBR-1:"])
    for row in comparison.sort_values("dataset").itertuples(index=False):
        mape_marker = "[✓]" if row.beats_paper_MAPE else "[✗]"
        r2_marker = "[✓]" if row.beats_or_ties_paper_R2 else "[✗]"
        lines.append(
            f"{row.dataset}: {mape_marker} MAPE {row.MAPE:.6f} vs paper {row.paper_MAPE:.6f}; "
            f"{r2_marker} R2 {row.R2:.6f} vs paper {row.paper_R2:.6f}; "
            f"MAE={row.MAE:.6f}; RMSE={row.RMSE:.6f}"
        )
    (OUTPUT_DIR / "all_dataset_run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_report(availability: pd.DataFrame, comparison: pd.DataFrame) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>All Paper Dataset Comparison</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 32px; background: #f7f9fb; color: #1f2933; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; margin: 16px 0 28px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    th, td {{ border-bottom: 1px solid #dde3ea; padding: 8px 9px; font-size: 13px; text-align: left; }}
    th {{ background: #e8eef5; }}
    .status {{ display: inline-block; min-width: 72px; padding: 4px 8px; border-radius: 999px; color: #fff; font-weight: 700; text-align: center; }}
    .win {{ background: #168a4a; }}
    .lose {{ background: #c72c2c; }}
  </style>
</head>
<body>
  <h1>All Paper Dataset Comparison</h1>
  <h2>Dataset Availability</h2>
  {availability_html(availability)}
  <h2>Available Results vs Paper PredXGBR-1</h2>
  {comparison_html(comparison)}
</body>
</html>
"""
    (OUTPUT_DIR / "all_dataset_comparison_stickers.html").write_text(html, encoding="utf-8")


def availability_html(availability: pd.DataFrame) -> str:
    rows = []
    for row in availability.itertuples(index=False):
        cls = "win" if row.available else "lose"
        label = "✓ FOUND" if row.available else "✗ MISSING"
        rows.append(f"<tr><td>{row.dataset}</td><td><span class='status {cls}'>{label}</span></td><td>{row.expected_file}</td></tr>")
    return "<table><thead><tr><th>Dataset</th><th>Status</th><th>Expected file</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def comparison_html(comparison: pd.DataFrame) -> str:
    rows = []
    for row in comparison.sort_values("dataset").itertuples(index=False):
        mape_cls = "win" if row.beats_paper_MAPE else "lose"
        r2_cls = "win" if row.beats_or_ties_paper_R2 else "lose"
        rows.append(
            "<tr>"
            f"<td>{row.dataset}</td><td>{row.MAPE:.6f}</td><td>{row.paper_MAPE:.6f}</td>"
            f"<td><span class='status {mape_cls}'>{'✓ WIN' if row.beats_paper_MAPE else '✗ LOSE'}</span></td>"
            f"<td>{row.R2:.6f}</td><td>{row.paper_R2:.6f}</td>"
            f"<td><span class='status {r2_cls}'>{'✓ WIN' if row.beats_or_ties_paper_R2 else '✗ LOSE'}</span></td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Dataset</th><th>Local MAPE</th><th>Paper MAPE</th><th>MAPE Status</th><th>Local R2</th><th>Paper R2</th><th>R2 Status</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def plot_comparison(comparison: pd.DataFrame) -> None:
    if comparison.empty:
        return
    ordered = comparison.sort_values("dataset")
    x = np.arange(len(ordered))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, ordered["MAPE"], width, label="Local MAPE")
    plt.bar(x + width / 2, ordered["paper_MAPE"], width, label="Paper PredXGBR-1 MAPE")
    plt.xticks(x, ordered["dataset"])
    plt.ylabel("MAPE (%)")
    plt.title("Available Datasets vs Paper PredXGBR-1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_dataset_mape_vs_paper.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
