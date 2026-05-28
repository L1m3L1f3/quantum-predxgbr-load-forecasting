from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline_pipeline import DATASETS, RESULTS_DIR, load_dataset, paths_for, split_features


OUTPUT_DIR = RESULTS_DIR / "postprocess_baseline_sweep"


@dataclass
class PostprocessConfig:
    dataset: str
    split_date: str
    feature_mode: str
    validation_fraction: float
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep fair post-processors around the trained XGBoost baseline.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PJME")
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--split-date", default=None)
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="causal")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def metric_row(model: str, y_true: pd.Series, pred: np.ndarray, seconds: float, notes: str) -> dict:
    return {
        "model": model,
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "MAPE": np.mean(np.abs((y_true.to_numpy() - pred) / y_true.to_numpy())) * 100,
        "R2": r2_score(y_true, pred),
        "seconds": seconds,
        "notes": notes,
    }


def run_postprocess_sweep(args: argparse.Namespace) -> pd.DataFrame:
    dataset = args.dataset.upper()
    split_date = args.split_date or DATASETS[dataset]["split_date"]
    df, _, _, _ = load_dataset(dataset, args.data_file)
    X_train, y_train, X_test, y_test, train, test = split_features(df, split_date, args.feature_mode)

    model_path = paths_for(dataset)["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing XGBoost baseline model: {model_path}. Run .venv/bin/python run_all.py first.")
    with model_path.open("rb") as handle:
        baseline_model = pickle.load(handle)

    baseline_train_pred = baseline_model.predict(X_train)
    baseline_test_pred = baseline_model.predict(X_test)
    train_residual = y_train.to_numpy() - baseline_train_pred

    split_at = int(len(X_train) * (1.0 - args.validation_fraction))
    X_fit = X_train.iloc[:split_at]
    y_fit = y_train.iloc[:split_at]
    fit_pred = baseline_train_pred[:split_at]
    fit_residual = train_residual[:split_at]
    X_val = X_train.iloc[split_at:]
    y_val = y_train.iloc[split_at:]
    val_pred = baseline_train_pred[split_at:]
    val_residual = train_residual[split_at:]

    rows = [metric_row("xgboost_baseline", y_test, baseline_test_pred, 0.0, "original trained baseline")]
    predictions = {"xgboost_baseline": baseline_test_pred}
    validation_scores = []

    variants = [
        fit_affine_calibration(fit_pred, y_fit, val_pred, y_val, baseline_test_pred),
        fit_hourly_bias(train.iloc[:split_at], fit_residual, train.iloc[split_at:], val_pred, test, baseline_test_pred, y_val),
        fit_residual_ridge(X_fit, fit_pred, fit_residual, X_val, val_pred, y_val, X_test, baseline_test_pred),
        fit_residual_xgboost(X_fit, fit_pred, fit_residual, X_val, val_pred, y_val, X_test, baseline_test_pred, args.random_state),
        fit_sequential_residual(train_residual, val_residual, y_val, val_pred, y_test, baseline_test_pred),
    ]

    start = time.perf_counter()
    best_variant = min(variants, key=lambda item: item["validation_MAE"])
    selected_pred = best_variant["test_pred"]
    rows.append(
        metric_row(
            "validation_selected_postprocessor",
            y_test,
            selected_pred,
            time.perf_counter() - start,
            f"selected {best_variant['model']} by validation MAE",
        )
    )
    predictions["validation_selected_postprocessor"] = selected_pred

    for variant in variants:
        rows.append(metric_row(variant["model"], y_test, variant["test_pred"], variant["seconds"], variant["notes"]))
        rows[-1]["validation_MAE"] = variant["validation_MAE"]
        predictions[variant["model"]] = variant["test_pred"]
        validation_scores.append({"model": variant["model"], "validation_MAE": variant["validation_MAE"], "notes": variant["notes"]})

    metrics = pd.DataFrame(rows)
    metrics.insert(0, "dataset", dataset)
    metrics["beats_baseline"] = metrics["MAE"] < float(metrics.loc[metrics["model"] == "xgboost_baseline", "MAE"].iloc[0])
    metrics["MAE_improvement_vs_baseline"] = float(metrics.loc[metrics["model"] == "xgboost_baseline", "MAE"].iloc[0]) - metrics["MAE"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "postprocess_baseline_sweep_metrics.csv", index=False)
    pd.DataFrame(validation_scores).to_csv(OUTPUT_DIR / "postprocess_validation_scores.csv", index=False)
    save_predictions(test.index, y_test, predictions)
    plot_metrics(metrics)
    plot_predictions(test.index, y_test, baseline_test_pred, selected_pred)
    config = PostprocessConfig(dataset, split_date, args.feature_mode, args.validation_fraction, args.random_state)
    (OUTPUT_DIR / "postprocess_baseline_sweep_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    print(metrics.sort_values("MAE").to_string(index=False))
    return metrics


def fit_affine_calibration(fit_pred, y_fit, val_pred, y_val, test_pred) -> dict:
    start = time.perf_counter()
    model = LinearRegression()
    model.fit(fit_pred.reshape(-1, 1), y_fit)
    val_out = model.predict(val_pred.reshape(-1, 1))
    test_out = model.predict(test_pred.reshape(-1, 1))
    return {
        "model": "affine_prediction_calibration",
        "validation_MAE": mean_absolute_error(y_val, val_out),
        "test_pred": test_out,
        "seconds": time.perf_counter() - start,
        "notes": "linear calibration y = a * baseline_prediction + b",
    }


def fit_hourly_bias(train_fit: pd.DataFrame, fit_residual: np.ndarray, train_val: pd.DataFrame, val_pred, test: pd.DataFrame, test_pred, y_val) -> dict:
    start = time.perf_counter()
    bias = pd.Series(fit_residual, index=train_fit.index).groupby(train_fit["hour"]).mean()
    global_bias = float(np.mean(fit_residual))
    val_bias = train_val["hour"].map(bias).fillna(global_bias).to_numpy()
    test_bias = test["hour"].map(bias).fillna(global_bias).to_numpy()
    val_out = val_pred + val_bias
    test_out = test_pred + test_bias
    return {
        "model": "hourly_residual_bias",
        "validation_MAE": mean_absolute_error(y_val, val_out),
        "test_pred": test_out,
        "seconds": time.perf_counter() - start,
        "notes": "adds mean training residual by hour of day",
    }


def residual_features(X: pd.DataFrame, baseline_pred: np.ndarray) -> np.ndarray:
    return np.column_stack([X.to_numpy(), baseline_pred])


def fit_residual_ridge(X_fit, fit_pred, fit_residual, X_val, val_pred, y_val, X_test, test_pred) -> dict:
    start = time.perf_counter()
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    model.fit(residual_features(X_fit, fit_pred), fit_residual)
    val_out = val_pred + model.predict(residual_features(X_val, val_pred))
    test_out = test_pred + model.predict(residual_features(X_test, test_pred))
    return {
        "model": "residual_ridge_features",
        "validation_MAE": mean_absolute_error(y_val, val_out),
        "test_pred": test_out,
        "seconds": time.perf_counter() - start,
        "notes": "ridge model predicts baseline residual from causal features",
    }


def fit_residual_xgboost(X_fit, fit_pred, fit_residual, X_val, val_pred, y_val, X_test, test_pred, random_state: int) -> dict:
    start = time.perf_counter()
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(
        residual_features(X_fit, fit_pred),
        fit_residual,
        eval_set=[(residual_features(X_val, val_pred), y_val.to_numpy() - val_pred)],
        verbose=False,
    )
    val_out = val_pred + model.predict(residual_features(X_val, val_pred))
    test_out = test_pred + model.predict(residual_features(X_test, test_pred))
    return {
        "model": "residual_xgboost_features",
        "validation_MAE": mean_absolute_error(y_val, val_out),
        "test_pred": test_out,
        "seconds": time.perf_counter() - start,
        "notes": "small XGBoost model predicts baseline residual from causal features",
    }


def fit_sequential_residual(train_residual: np.ndarray, val_residual: np.ndarray, y_val, val_pred, y_test, test_pred) -> dict:
    start = time.perf_counter()
    lag = train_residual[:-1]
    target = train_residual[1:]
    denominator = float(np.dot(lag, lag))
    alpha = 0.0 if denominator == 0.0 else float(np.dot(lag, target) / denominator)
    val_prev = np.r_[train_residual[-1], val_residual[:-1]]
    val_out = val_pred + alpha * val_prev
    test_residual = y_test.to_numpy() - test_pred
    test_prev = np.r_[val_residual[-1], test_residual[:-1]]
    test_out = test_pred + alpha * test_prev
    return {
        "model": "sequential_last_residual_correction",
        "validation_MAE": mean_absolute_error(y_val, val_out),
        "test_pred": test_out,
        "seconds": time.perf_counter() - start,
        "notes": f"one-step correction using previous observed residual, alpha={alpha:.4f}",
    }


def save_predictions(index: pd.Index, actual: pd.Series, predictions: dict[str, np.ndarray]) -> None:
    data = {"datetime": index, "actual_load": actual.to_numpy()}
    for model, pred in predictions.items():
        data[model] = pred
    pd.DataFrame(data).to_csv(OUTPUT_DIR / "postprocess_baseline_sweep_predictions.csv", index=False)


def plot_metrics(metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("MAE", ascending=True)
    colors = ["#2a9d8f" if row else "#4e79a7" for row in ordered["beats_baseline"]]
    plt.figure(figsize=(11, 6))
    plt.barh(ordered["model"][::-1], ordered["MAE"].to_numpy()[::-1], color=colors[::-1])
    plt.xlabel("MAE lower is better")
    plt.title("Baseline Post-Processing Sweep")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "postprocess_baseline_sweep_mae.png", dpi=150)
    plt.close()


def plot_predictions(index: pd.Index, actual: pd.Series, baseline_pred: np.ndarray, selected_pred: np.ndarray) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(index, actual, label="Actual", linewidth=1)
    plt.plot(index, baseline_pred, label="XGBoost baseline", linewidth=1)
    plt.plot(index, selected_pred, label="Selected postprocessor", linewidth=1)
    plt.title("Selected Postprocessor vs XGBoost Baseline")
    plt.xlabel("Datetime")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "postprocess_selected_vs_baseline.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_postprocess_sweep(parse_args())
