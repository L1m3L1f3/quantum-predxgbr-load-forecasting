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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from baseline_pipeline import DATASETS, MODEL_DIR, RESULTS_DIR, load_dataset, paths_for, split_features


OUTPUT_DIR = RESULTS_DIR / "hybrid_quantum_residual"
MODEL_OUTPUT_DIR = MODEL_DIR / "hybrid_quantum_residual"


@dataclass
class HybridResidualConfig:
    dataset: str
    split_date: str
    feature_mode: str
    device_name: str
    n_qubits: int
    n_layers: int
    max_train_samples: int
    test_rows: int
    readout: str
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an XGBoost plus GPU quantum residual baseline.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PJME")
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--split-date", default=None)
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="causal")
    parser.add_argument("--device", default="lightning.gpu")
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--readout", choices=["ridge", "xgboost"], default="xgboost")
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


def sample_train_rows(X: pd.DataFrame, y: pd.Series, max_samples: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    if max_samples <= 0 or len(X) <= max_samples:
        return X, y
    sampled_index = X.sample(n=max_samples, random_state=random_state).sort_index().index
    return X.loc[sampled_index], y.loc[sampled_index]


def build_readout(kind: str, random_state: int):
    if kind == "ridge":
        return Ridge(alpha=1.0)
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=random_state,
        n_jobs=-1,
    )


def metric_row(name: str, y_true: pd.Series, predictions: np.ndarray, training_time: float | None = None) -> dict[str, float | str]:
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "MAPE": np.mean(np.abs((y_true.to_numpy() - predictions) / y_true.to_numpy())) * 100,
        "R2": r2_score(y_true, predictions),
        "training_time_seconds": np.nan if training_time is None else training_time,
    }


def run_hybrid_residual(args: argparse.Namespace) -> pd.DataFrame:
    dataset = args.dataset.upper()
    split_date = args.split_date or DATASETS[dataset]["split_date"]
    df, _, _, _ = load_dataset(dataset, args.data_file)
    X_train, y_train, X_test, y_test, _, test = split_features(df, split_date, args.feature_mode)

    xgb_model_path = paths_for(dataset)["model"]
    if not xgb_model_path.exists():
        raise FileNotFoundError(f"Missing XGBoost baseline model: {xgb_model_path}. Run .venv/bin/python run_all.py first.")
    with xgb_model_path.open("rb") as handle:
        baseline_model = pickle.load(handle)

    baseline_train_pred = baseline_model.predict(X_train)
    baseline_test_pred = baseline_model.predict(X_test)
    train_residual = y_train.to_numpy() - baseline_train_pred

    X_residual_train, residual_train = sample_train_rows(
        X_train,
        pd.Series(train_residual, index=X_train.index),
        args.max_train_samples,
        args.random_state,
    )
    baseline_residual_train_pred = baseline_model.predict(X_residual_train)

    preprocessing = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=args.n_qubits, random_state=args.random_state)),
            ("angle_scale", MinMaxScaler(feature_range=(-np.pi, np.pi))),
        ]
    )
    X_train_angles = preprocessing.fit_transform(X_residual_train)
    X_test_angles = preprocessing.transform(X_test)

    rng = np.random.default_rng(args.random_state)
    weights = rng.uniform(low=-np.pi, high=np.pi, size=(args.n_layers, args.n_qubits, 3))
    circuit = make_circuit(args.n_qubits, weights, args.device)

    start = time.perf_counter()
    Z_train = quantum_transform(X_train_angles, circuit)
    Z_test = quantum_transform(X_test_angles, circuit)

    readout_train = np.column_stack([Z_train, baseline_residual_train_pred])
    readout_test = np.column_stack([Z_test, baseline_test_pred])
    readout = build_readout(args.readout, args.random_state)

    if args.readout == "xgboost":
        Z_fit, Z_val, r_fit, r_val = train_test_split(
            readout_train,
            residual_train.to_numpy(),
            test_size=0.2,
            shuffle=False,
        )
        readout.fit(Z_fit, r_fit, eval_set=[(Z_val, r_val)], verbose=False)
        val_baseline = y_train.loc[X_residual_train.index].to_numpy()[-len(r_val) :] - r_val
        val_true = y_train.loc[X_residual_train.index].to_numpy()[-len(r_val) :]
        val_residual_pred = readout.predict(Z_val)
    else:
        Z_fit, Z_val, r_fit, r_val = train_test_split(
            readout_train,
            residual_train.to_numpy(),
            test_size=0.2,
            shuffle=False,
        )
        readout.fit(Z_fit, r_fit)
        val_baseline = y_train.loc[X_residual_train.index].to_numpy()[-len(r_val) :] - r_val
        val_true = y_train.loc[X_residual_train.index].to_numpy()[-len(r_val) :]
        val_residual_pred = readout.predict(Z_val)

    residual_test_pred = readout.predict(readout_test)
    candidate_alphas = np.linspace(-1.0, 1.0, 81)
    alpha_scores = [
        mean_squared_error(val_true, val_baseline + alpha * val_residual_pred)
        for alpha in candidate_alphas
    ]
    best_alpha = float(candidate_alphas[int(np.argmin(alpha_scores))])
    hybrid_pred = baseline_test_pred + best_alpha * residual_test_pred
    training_time = time.perf_counter() - start

    metrics = pd.DataFrame(
        [
            metric_row("xgboost_baseline", y_test, baseline_test_pred),
            metric_row(f"hybrid_quantum_residual_{args.readout}", y_test, hybrid_pred, training_time),
        ]
    )
    metrics.insert(0, "dataset", dataset)
    metrics["residual_alpha"] = [np.nan, best_alpha]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "hybrid_quantum_residual_metrics.csv", index=False)
    pd.DataFrame(
        {
            "datetime": test.index,
            "actual_load": y_test.to_numpy(),
            "xgboost_prediction": baseline_test_pred,
            "hybrid_prediction": hybrid_pred,
            "quantum_residual_prediction": residual_test_pred,
        }
    ).to_csv(OUTPUT_DIR / "hybrid_quantum_residual_predictions.csv", index=False)

    config = HybridResidualConfig(
        dataset=dataset,
        split_date=split_date,
        feature_mode=args.feature_mode,
        device_name=args.device,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        max_train_samples=args.max_train_samples,
        test_rows=len(X_test),
        readout=args.readout,
        random_state=args.random_state,
    )
    (OUTPUT_DIR / "hybrid_quantum_residual_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    with (MODEL_OUTPUT_DIR / f"hybrid_quantum_residual_{dataset.lower()}.pkl").open("wb") as handle:
        pickle.dump(
            {
                "preprocessing": preprocessing,
                "weights": weights,
                "readout": readout,
                "config": asdict(config),
            },
            handle,
        )

    plot_comparison(test.index, y_test, baseline_test_pred, hybrid_pred, OUTPUT_DIR / "hybrid_quantum_residual_plot.png", dataset)
    print(metrics.to_string(index=False))
    return metrics


def plot_comparison(
    index: pd.Index,
    actual: pd.Series,
    baseline_pred: np.ndarray,
    hybrid_pred: np.ndarray,
    output_path: Path,
    dataset: str,
) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(index, actual, label="Actual", linewidth=1)
    plt.plot(index, baseline_pred, label="XGBoost", linewidth=1)
    plt.plot(index, hybrid_pred, label="XGBoost + quantum residual", linewidth=1)
    plt.title(f"{dataset} Hybrid Quantum Residual Comparison")
    plt.xlabel("Datetime")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
