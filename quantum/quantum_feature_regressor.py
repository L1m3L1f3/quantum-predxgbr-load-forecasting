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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from baseline_pipeline import DATASETS, RESULTS_DIR, load_dataset, split_features


QUANTUM_RESULTS_DIR = RESULTS_DIR / "quantum"
QUANTUM_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "quantum"


@dataclass
class QuantumRunConfig:
    dataset: str
    split_date: str
    feature_mode: str
    n_qubits: int
    n_layers: int
    device_name: str
    max_train_samples: int
    max_test_samples: int
    random_state: int
    train_rows_used: int
    test_rows_used: int
    model_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small hybrid quantum feature baseline.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PJME")
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--split-date", default=None)
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="causal")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--device", default="lightning.gpu", help="PennyLane device, e.g. lightning.gpu or default.qubit.")
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--max-test-samples", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
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


def subsample(
    X: pd.DataFrame,
    y: pd.Series,
    max_samples: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_samples <= 0 or len(X) <= max_samples:
        return X, y
    sampled_index = X.sample(n=max_samples, random_state=random_state).sort_index().index
    return X.loc[sampled_index], y.loc[sampled_index]


def run_quantum_baseline(args: argparse.Namespace) -> pd.DataFrame:
    dataset = args.dataset.upper()
    split_date = args.split_date or DATASETS[dataset]["split_date"]
    df, _, _, _ = load_dataset(dataset, args.data_file)
    X_train, y_train, X_test, y_test, _, test = split_features(df, split_date, args.feature_mode)

    X_train, y_train = subsample(X_train, y_train, args.max_train_samples, args.random_state)
    X_test, y_test = subsample(X_test, y_test, args.max_test_samples, args.random_state)
    test = test.loc[X_test.index]

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
    readout = Ridge(alpha=args.ridge_alpha)
    readout.fit(Z_train, y_train)
    predictions = readout.predict(Z_test)
    training_time = time.perf_counter() - start

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    metrics = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": "hybrid_quantum_feature_ridge",
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": rmse,
                "MAPE": np.mean(np.abs((y_test.to_numpy() - predictions) / y_test.to_numpy())) * 100,
                "R2": r2_score(y_test, predictions),
                "training_time_seconds": training_time,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_qubits": args.n_qubits,
                "n_layers": args.n_layers,
            }
        ]
    )

    QUANTUM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    QUANTUM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(QUANTUM_RESULTS_DIR / "quantum_metrics.csv", index=False)
    pd.DataFrame(
        {
            "datetime": test.index,
            "actual_load": y_test.to_numpy(),
            "predicted_load": predictions,
        }
    ).to_csv(QUANTUM_RESULTS_DIR / "quantum_predictions.csv", index=False)

    config = QuantumRunConfig(
        dataset=dataset,
        split_date=split_date,
        feature_mode=args.feature_mode,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        device_name=args.device,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        random_state=args.random_state,
        train_rows_used=len(X_train),
        test_rows_used=len(X_test),
        model_type="hybrid_quantum_feature_ridge",
    )
    (QUANTUM_RESULTS_DIR / "quantum_run_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    with (QUANTUM_MODEL_DIR / f"hybrid_quantum_feature_{dataset.lower()}.pkl").open("wb") as handle:
        pickle.dump(
            {
                "preprocessing": preprocessing,
                "readout": readout,
                "weights": weights,
                "config": asdict(config),
            },
            handle,
        )

    plot_predictions(test.index, y_test, predictions, QUANTUM_RESULTS_DIR / "quantum_actual_vs_predicted.png", dataset)
    print(metrics.to_string(index=False))
    return metrics


def plot_predictions(index: pd.Index, actual: pd.Series, predicted: np.ndarray, output_path: Path, dataset: str) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(index, actual, label="Actual", linewidth=1)
    plt.plot(index, predicted, label="Hybrid quantum", linewidth=1)
    plt.title(f"{dataset} Hybrid Quantum Feature Baseline")
    plt.xlabel("Datetime")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    run_quantum_baseline(parse_args())
