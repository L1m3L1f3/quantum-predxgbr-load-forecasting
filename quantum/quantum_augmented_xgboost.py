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
import pennylane as qml
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from baseline_pipeline import DATASETS, MODEL_DIR, RESULTS_DIR, load_dataset, paths_for, split_features


OUTPUT_DIR = RESULTS_DIR / "quantum_augmented_xgboost"
MODEL_OUTPUT_DIR = MODEL_DIR / "quantum_augmented_xgboost"


@dataclass
class QuantumAugmentedConfig:
    dataset: str
    split_date: str
    feature_mode: str
    device_name: str
    n_qubits: int
    n_layers: int
    train_rows_used: int
    test_rows_used: int
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost with additional GPU quantum features.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PJME")
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--split-date", default=None)
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="causal")
    parser.add_argument("--device", default="lightning.gpu")
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-test-samples", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def make_circuit(n_qubits: int, weights: np.ndarray, device_name: str):
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x: np.ndarray):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def quantum_transform(X: np.ndarray, circuit) -> np.ndarray:
    return np.asarray([circuit(row) for row in X], dtype=float)


def sample_rows(X: pd.DataFrame, y: pd.Series, max_samples: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    if max_samples <= 0 or len(X) <= max_samples:
        return X, y
    index = X.sample(n=max_samples, random_state=random_state).sort_index().index
    return X.loc[index], y.loc[index]


def metric_row(name: str, y_true: pd.Series, predictions: np.ndarray, training_time: float | None = None) -> dict[str, float | str]:
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "MAPE": np.mean(np.abs((y_true.to_numpy() - predictions) / y_true.to_numpy())) * 100,
        "R2": r2_score(y_true, predictions),
        "training_time_seconds": np.nan if training_time is None else training_time,
    }


def run_quantum_augmented_xgboost(args: argparse.Namespace) -> pd.DataFrame:
    dataset = args.dataset.upper()
    split_date = args.split_date or DATASETS[dataset]["split_date"]
    df, _, _, _ = load_dataset(dataset, args.data_file)
    X_train, y_train, X_test, y_test, _, test = split_features(df, split_date, args.feature_mode)

    X_train, y_train = sample_rows(X_train, y_train, args.max_train_samples, args.random_state)
    X_test, y_test = sample_rows(X_test, y_test, args.max_test_samples, args.random_state)
    test = test.loc[X_test.index]

    baseline_model_path = paths_for(dataset)["model"]
    if not baseline_model_path.exists():
        raise FileNotFoundError(f"Missing XGBoost baseline model: {baseline_model_path}. Run .venv/bin/python run_all.py first.")
    with baseline_model_path.open("rb") as handle:
        baseline_model = pickle.load(handle)
    baseline_pred = baseline_model.predict(X_test)

    preprocessing = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=args.n_qubits, random_state=args.random_state)),
            ("angle_scale", MinMaxScaler(feature_range=(-np.pi, np.pi))),
        ]
    )
    X_train_angles = preprocessing.fit_transform(X_train)
    X_test_angles = preprocessing.transform(X_test)

    rng = np.random.default_rng(args.random_state)
    weights = rng.uniform(low=-np.pi, high=np.pi, size=(args.n_layers, args.n_qubits, 3))
    circuit = make_circuit(args.n_qubits, weights, args.device)

    start = time.perf_counter()
    Z_train = quantum_transform(X_train_angles, circuit)
    Z_test = quantum_transform(X_test_angles, circuit)
    X_train_aug = np.column_stack([X_train.to_numpy(), Z_train])
    X_test_aug = np.column_stack([X_test.to_numpy(), Z_test])

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=args.random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(X_train_aug, y_train, eval_set=[(X_train_aug, y_train), (X_test_aug, y_test)], verbose=False)
    quantum_augmented_pred = model.predict(X_test_aug)
    training_time = time.perf_counter() - start

    metrics = pd.DataFrame(
        [
            metric_row("xgboost_baseline_same_rows", y_test, baseline_pred),
            metric_row("quantum_augmented_xgboost", y_test, quantum_augmented_pred, training_time),
        ]
    )
    metrics.insert(0, "dataset", dataset)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "quantum_augmented_xgboost_metrics.csv", index=False)
    pd.DataFrame(
        {
            "datetime": test.index,
            "actual_load": y_test.to_numpy(),
            "xgboost_prediction": baseline_pred,
            "quantum_augmented_prediction": quantum_augmented_pred,
        }
    ).to_csv(OUTPUT_DIR / "quantum_augmented_xgboost_predictions.csv", index=False)

    config = QuantumAugmentedConfig(
        dataset=dataset,
        split_date=split_date,
        feature_mode=args.feature_mode,
        device_name=args.device,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        train_rows_used=len(X_train),
        test_rows_used=len(X_test),
        random_state=args.random_state,
    )
    (OUTPUT_DIR / "quantum_augmented_xgboost_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    with (MODEL_OUTPUT_DIR / f"quantum_augmented_xgboost_{dataset.lower()}.pkl").open("wb") as handle:
        pickle.dump({"model": model, "preprocessing": preprocessing, "weights": weights, "config": asdict(config)}, handle)

    plot_comparison(test.index, y_test, baseline_pred, quantum_augmented_pred, OUTPUT_DIR / "quantum_augmented_xgboost_plot.png", dataset)
    print(metrics.to_string(index=False))
    return metrics


def plot_comparison(index: pd.Index, actual: pd.Series, baseline: np.ndarray, augmented: np.ndarray, output_path: Path, dataset: str) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(index, actual, label="Actual", linewidth=1)
    plt.plot(index, baseline, label="XGBoost baseline", linewidth=1)
    plt.plot(index, augmented, label="Quantum-augmented XGBoost", linewidth=1)
    plt.title(f"{dataset} Quantum-Augmented XGBoost Comparison")
    plt.xlabel("Datetime")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
