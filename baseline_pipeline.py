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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "processed"


DATASETS = {
    "PJME": {"file": "PJME_hourly.csv", "target_candidates": ["PJME_MW", "Load"], "split_date": "2015-01-02"},
    "PJM": {"file": "PJM_Load_hourly.csv", "target_candidates": ["PJM_Load_MW", "Load"], "split_date": "2000-08-07"},
    "AEP": {"file": "AEP_hourly.csv", "target_candidates": ["AEP_MW", "Load"], "split_date": "2015-05-28"},
    "DAYTON": {"file": "DAYTON_hourly.csv", "target_candidates": ["DAYTON_MW", "Load"], "split_date": "2015-05-28"},
    "PJMW": {"file": "PJMW_hourly.csv", "target_candidates": ["PJMW_MW", "Load"], "split_date": "2015-01-02"},
}


FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "quarter",
    "month",
    "year",
    "dayofyear",
    "dayofmonth",
    "weekofyear",
    "load_6_hrs_lag",
    "load_12_hrs_lag",
    "load_24_hrs_lag",
    "load_6_hrs_mean",
    "load_12_hrs_mean",
    "load_24_hrs_mean",
    "load_6_hrs_std",
    "load_12_hrs_std",
    "load_24_hrs_std",
    "load_6_hrs_max",
    "load_12_hrs_max",
    "load_24_hrs_max",
    "load_6_hrs_min",
    "load_12_hrs_min",
    "load_24_hrs_min",
]


@dataclass
class RunConfig:
    dataset: str
    data_file: str
    datetime_column: str
    target_column: str
    split_date: str
    forecasting_horizon_hours: int
    train_rows: int
    test_rows: int
    feature_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PredXGBR classical load forecasting baseline.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PJME")
    parser.add_argument("--data-file", type=Path, default=None, help="Optional CSV path. Defaults to data/<dataset>_hourly.csv.")
    parser.add_argument("--split-date", default=None, help="Date boundary for train/test split.")
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="causal")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def paths_for(dataset: str) -> dict[str, Path]:
    stem = dataset.lower()
    return {
        "model": MODEL_DIR / f"xgboost_{stem}.pkl",
        "config": RESULTS_DIR / "run_config.json",
        "training_metadata": RESULTS_DIR / "training_metadata.json",
        "metrics": RESULTS_DIR / "metrics.csv",
        "predictions": RESULTS_DIR / "predictions.csv",
        "history": RESULTS_DIR / "training_history.csv",
        "actual_plot": RESULTS_DIR / "actual_vs_predicted.png",
        "loss_plot": RESULTS_DIR / "loss_curve.png",
        "train_features": PROCESSED_DIR / f"{stem}_train_features.csv",
        "test_features": PROCESSED_DIR / f"{stem}_test_features.csv",
    }


def resolve_data_file(dataset: str, data_file: Path | None) -> Path:
    if data_file is not None:
        return data_file
    return DATA_DIR / DATASETS[dataset]["file"]


def load_dataset(dataset: str, data_file: Path | None = None) -> tuple[pd.DataFrame, str, str, Path]:
    path = resolve_data_file(dataset, data_file)
    if not path.exists():
        expected = DATASETS[dataset]["file"]
        raise FileNotFoundError(
            f"Missing dataset file: {path}\n"
            f"Place {expected} in {DATA_DIR}, or pass --data-file /path/to/{expected}."
        )

    df = pd.read_csv(path)
    datetime_column = df.columns[0]
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df = df.sort_values(datetime_column).set_index(datetime_column)

    target_column = next((col for col in DATASETS[dataset]["target_candidates"] if col in df.columns), None)
    if target_column is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) != 1:
            raise ValueError(
                f"Could not infer target column. Expected one of {DATASETS[dataset]['target_candidates']} "
                f"or exactly one numeric column; found {numeric_columns}."
            )
        target_column = numeric_columns[0]

    df = df[[target_column]].rename(columns={target_column: "Load"})
    return df, datetime_column, target_column, path


def create_features(df: pd.DataFrame, feature_mode: str = "causal") -> pd.DataFrame:
    features = df.copy()
    dates = features.index
    features["hour"] = dates.hour
    features["dayofweek"] = dates.dayofweek
    features["quarter"] = dates.quarter
    features["month"] = dates.month
    features["year"] = dates.year
    features["dayofyear"] = dates.dayofyear
    features["dayofmonth"] = dates.day
    features["weekofyear"] = dates.isocalendar().week.astype(int)

    load_source = features["Load"].shift(1) if feature_mode == "causal" else features["Load"]
    for window in (6, 12, 24):
        features[f"load_{window}_hrs_lag"] = features["Load"].shift(window)
        rolling = load_source.rolling(window=window)
        features[f"load_{window}_hrs_mean"] = rolling.mean()
        features[f"load_{window}_hrs_std"] = rolling.std()
        features[f"load_{window}_hrs_max"] = rolling.max()
        features[f"load_{window}_hrs_min"] = rolling.min()

    return features


def split_features(
    df: pd.DataFrame, split_date: str, feature_mode: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    featured = create_features(df, feature_mode=feature_mode)
    train = featured.loc[featured.index <= split_date].copy()
    test = featured.loc[featured.index > split_date].copy()

    # XGBoost can consume NaN values, but dropping the first rows makes metrics easier to compare across models.
    train = train.dropna(subset=FEATURE_COLUMNS + ["Load"])
    test = test.dropna(subset=FEATURE_COLUMNS + ["Load"])
    return train[FEATURE_COLUMNS], train["Load"], test[FEATURE_COLUMNS], test["Load"], train, test


def preprocess(args: argparse.Namespace) -> RunConfig:
    dataset = args.dataset.upper()
    split_date = args.split_date or DATASETS[dataset]["split_date"]
    df, datetime_column, target_column, data_path = load_dataset(dataset, args.data_file)
    X_train, y_train, X_test, y_test, train, test = split_features(df, split_date, args.feature_mode)

    PROCESSED_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    train.assign(Load=y_train).to_csv(paths_for(dataset)["train_features"])
    test.assign(Load=y_test).to_csv(paths_for(dataset)["test_features"])

    config = RunConfig(
        dataset=dataset,
        data_file=str(data_path),
        datetime_column=datetime_column,
        target_column=target_column,
        split_date=split_date,
        forecasting_horizon_hours=1,
        train_rows=len(X_train),
        test_rows=len(X_test),
        feature_mode=args.feature_mode,
    )
    paths_for(dataset)["config"].write_text(json.dumps(asdict(config), indent=2) + "\n")
    print(f"Preprocessed {dataset}: {len(X_train)} train rows, {len(X_test)} test rows")
    return config


def train(args: argparse.Namespace) -> float:
    dataset = args.dataset.upper()
    split_date = args.split_date or DATASETS[dataset]["split_date"]
    df, _, _, _ = load_dataset(dataset, args.data_file)
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
    training_time = time.perf_counter() - start

    MODEL_DIR.mkdir(exist_ok=True)
    with paths_for(dataset)["model"].open("wb") as handle:
        pickle.dump(model, handle)

    history = model.evals_result()
    history_df = pd.DataFrame(
        {
            "iteration": range(len(history["validation_0"]["rmse"])),
            "train_rmse": history["validation_0"]["rmse"],
            "test_rmse": history["validation_1"]["rmse"],
        }
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    history_df.to_csv(paths_for(dataset)["history"], index=False)
    paths_for(dataset)["training_metadata"].write_text(
        json.dumps({"dataset": dataset, "training_time_seconds": training_time}, indent=2) + "\n"
    )
    print(f"Trained {dataset} XGBoost model in {training_time:.2f} seconds")
    return training_time


def evaluate(args: argparse.Namespace, training_time: float | None = None) -> pd.DataFrame:
    dataset = args.dataset.upper()
    split_date = args.split_date or DATASETS[dataset]["split_date"]
    df, _, _, _ = load_dataset(dataset, args.data_file)
    _, _, X_test, y_test, _, test = split_features(df, split_date, args.feature_mode)

    model_path = paths_for(dataset)["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}. Run training first.")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    if training_time is None and paths_for(dataset)["training_metadata"].exists():
        metadata = json.loads(paths_for(dataset)["training_metadata"].read_text())
        training_time = metadata.get("training_time_seconds")
    metrics = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": rmse,
                "MAPE": np.mean(np.abs((y_test.to_numpy() - predictions) / y_test.to_numpy())) * 100,
                "R2": r2_score(y_test, predictions),
                "training_time_seconds": training_time if training_time is not None else np.nan,
            }
        ]
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(paths_for(dataset)["metrics"], index=False)
    pd.DataFrame(
        {
            "datetime": test.index,
            "actual_load": y_test.to_numpy(),
            "predicted_load": predictions,
        }
    ).to_csv(paths_for(dataset)["predictions"], index=False)

    plot_actual_vs_predicted(test.index, y_test, predictions, paths_for(dataset)["actual_plot"], dataset)
    plot_loss_curve(paths_for(dataset)["history"], paths_for(dataset)["loss_plot"])
    print(metrics.to_string(index=False))
    return metrics


def plot_actual_vs_predicted(index: pd.Index, actual: pd.Series, predicted: np.ndarray, output_path: Path, dataset: str) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(index, actual, label="Actual", linewidth=1)
    plt.plot(index, predicted, label="Predicted", linewidth=1)
    plt.title(f"{dataset} Actual vs Predicted Load")
    plt.xlabel("Datetime")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_loss_curve(history_path: Path, output_path: Path) -> None:
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    plt.figure(figsize=(9, 5))
    plt.plot(history["iteration"], history["train_rmse"], label="Train RMSE")
    plt.plot(history["iteration"], history["test_rmse"], label="Test RMSE")
    plt.title("XGBoost Training History")
    plt.xlabel("Boosting iteration")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_all(args: argparse.Namespace) -> None:
    preprocess(args)
    training_time = train(args)
    evaluate(args, training_time=training_time)


if __name__ == "__main__":
    run_all(parse_args())
