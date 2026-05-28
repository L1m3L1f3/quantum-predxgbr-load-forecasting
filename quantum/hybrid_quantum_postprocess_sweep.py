from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime
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

from baseline_pipeline import DATASETS, RESULTS_DIR, load_dataset, paths_for, split_features


OUTPUT_DIR = RESULTS_DIR / "hybrid_quantum_postprocess"


@dataclass
class HybridQuantumPostprocessConfig:
    dataset: str
    split_date: str
    feature_mode: str
    device_name: str
    n_qubits: int
    n_layers: int
    max_train_samples: int
    validation_fraction: float
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-test hybrid quantum residual post-processing sweep.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PJME")
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--split-date", default=None)
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="causal")
    parser.add_argument("--device", default="lightning.gpu")
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=Path("logs") / "hybrid_quantum_postprocess")
    return parser.parse_args()


def make_circuit(n_qubits: int, weights: np.ndarray, device_name: str):
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x: np.ndarray):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def quantum_transform(X: np.ndarray, circuit, log_path: Path, label: str) -> np.ndarray:
    rows = []
    total = len(X)
    checkpoint = max(1, total // 10)
    for index, row in enumerate(X, start=1):
        rows.append(circuit(row))
        if index == 1 or index == total or index % checkpoint == 0:
            log_message(log_path, f"{label}: quantum rows {index}/{total}")
    return np.asarray(rows, dtype=float)


def sample_train_rows(X: pd.DataFrame, y: pd.Series, max_samples: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    if max_samples <= 0 or len(X) <= max_samples:
        return X, y
    index = X.sample(n=max_samples, random_state=random_state).sort_index().index
    return X.loc[index], y.loc[index]


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


def run_hybrid_quantum_postprocess(args: argparse.Namespace) -> pd.DataFrame:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"hybrid_quantum_postprocess_{run_id}.log"
    log_message(log_path, f"Starting hybrid quantum postprocess run with args={serializable_args(args)}")

    dataset = args.dataset.upper()
    split_date = args.split_date or DATASETS[dataset]["split_date"]
    df, _, _, _ = load_dataset(dataset, args.data_file)
    X_train_full, y_train_full, X_test, y_test, _, test = split_features(df, split_date, args.feature_mode)

    model_path = paths_for(dataset)["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing XGBoost baseline model: {model_path}. Run .venv/bin/python run_all.py first.")
    with model_path.open("rb") as handle:
        baseline_model = pickle.load(handle)

    baseline_train_full = baseline_model.predict(X_train_full)
    baseline_test = baseline_model.predict(X_test)
    residual_full = y_train_full.to_numpy() - baseline_train_full

    X_residual, residual = sample_train_rows(
        X_train_full,
        pd.Series(residual_full, index=X_train_full.index),
        args.max_train_samples,
        args.random_state,
    )
    baseline_residual = baseline_model.predict(X_residual)
    y_residual_true = residual.to_numpy() + baseline_residual
    split_at = int(len(X_residual) * (1.0 - args.validation_fraction))

    X_fit = X_residual.iloc[:split_at]
    X_val = X_residual.iloc[split_at:]
    r_fit = residual.to_numpy()[:split_at]
    r_val = residual.to_numpy()[split_at:]
    base_fit = baseline_residual[:split_at]
    base_val = baseline_residual[split_at:]
    y_val = y_residual_true[split_at:]

    preprocessing = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=args.n_qubits, random_state=args.random_state)),
            ("angle_scale", MinMaxScaler(feature_range=(-np.pi, np.pi))),
        ]
    )
    X_fit_angles = preprocessing.fit_transform(X_fit)
    X_val_angles = preprocessing.transform(X_val)
    X_test_angles = preprocessing.transform(X_test)

    rng = np.random.default_rng(args.random_state)
    weights = rng.uniform(low=-np.pi, high=np.pi, size=(args.n_layers, args.n_qubits, 3))
    circuit = make_circuit(args.n_qubits, weights, args.device)

    start = time.perf_counter()
    Z_fit = quantum_transform(X_fit_angles, circuit, log_path, "fit")
    Z_val = quantum_transform(X_val_angles, circuit, log_path, "validation")
    Z_test = quantum_transform(X_test_angles, circuit, log_path, "test")

    quantum_readout = Ridge(alpha=1.0)
    quantum_readout.fit(np.column_stack([Z_fit, base_fit]), r_fit)
    q_val_residual = quantum_readout.predict(np.column_stack([Z_val, base_val]))
    q_test_residual = quantum_readout.predict(np.column_stack([Z_test, baseline_test]))

    classical_quantum_readout = Ridge(alpha=10.0)
    classical_quantum_readout.fit(np.column_stack([X_fit.to_numpy(), Z_fit, base_fit]), r_fit)
    hq_val_residual = classical_quantum_readout.predict(np.column_stack([X_val.to_numpy(), Z_val, base_val]))
    hq_test_residual = classical_quantum_readout.predict(np.column_stack([X_test.to_numpy(), Z_test, baseline_test]))

    val_prev = np.r_[residual.to_numpy()[split_at - 1], r_val[:-1]]
    test_residual_truth = y_test.to_numpy() - baseline_test
    test_prev = np.r_[r_val[-1], test_residual_truth[:-1]]

    rows = [metric_row("xgboost_baseline", y_test, baseline_test, 0.0, "original trained baseline")]
    predictions = {"xgboost_baseline": baseline_test}

    quantum_static = baseline_test + q_test_residual
    rows.append(metric_row("hybrid_quantum_residual_ridge_static", y_test, quantum_static, time.perf_counter() - start, "quantum residual ridge, static full-test prediction"))
    predictions["hybrid_quantum_residual_ridge_static"] = quantum_static

    hybrid_static = baseline_test + hq_test_residual
    rows.append(metric_row("hybrid_classical_quantum_residual_ridge_static", y_test, hybrid_static, time.perf_counter() - start, "classical plus quantum residual ridge, static full-test prediction"))
    predictions["hybrid_classical_quantum_residual_ridge_static"] = hybrid_static

    selected = select_blend(
        y_val=y_val,
        base_val=base_val,
        prev_val=val_prev,
        q_val=q_val_residual,
        hq_val=hq_val_residual,
        base_test=baseline_test,
        prev_test=test_prev,
        q_test=q_test_residual,
        hq_test=hq_test_residual,
    )
    rows.append(
        metric_row(
            "hybrid_quantum_sequential_validation_selected",
            y_test,
            selected["test_pred"],
            time.perf_counter() - start,
            selected["notes"],
        )
    )
    predictions["hybrid_quantum_sequential_validation_selected"] = selected["test_pred"]

    metrics = pd.DataFrame(rows)
    baseline_mae = float(metrics.loc[metrics["model"] == "xgboost_baseline", "MAE"].iloc[0])
    metrics.insert(0, "dataset", dataset)
    metrics["beats_baseline"] = metrics["MAE"] < baseline_mae
    metrics["MAE_improvement_vs_baseline"] = baseline_mae - metrics["MAE"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "hybrid_quantum_postprocess_metrics.csv", index=False)
    pd.DataFrame({"datetime": test.index, "actual_load": y_test.to_numpy(), **predictions}).to_csv(
        OUTPUT_DIR / "hybrid_quantum_postprocess_predictions.csv",
        index=False,
    )
    config = HybridQuantumPostprocessConfig(
        dataset=dataset,
        split_date=split_date,
        feature_mode=args.feature_mode,
        device_name=args.device,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        max_train_samples=args.max_train_samples,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
    )
    (OUTPUT_DIR / "hybrid_quantum_postprocess_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    plot_metrics(metrics)
    plot_predictions(test.index, y_test, baseline_test, selected["test_pred"])
    log_message(log_path, "Finished hybrid quantum postprocess run")
    log_message(log_path, metrics.sort_values("MAE").to_string(index=False))
    print(metrics.sort_values("MAE").to_string(index=False))
    return metrics


def select_blend(
    y_val: np.ndarray,
    base_val: np.ndarray,
    prev_val: np.ndarray,
    q_val: np.ndarray,
    hq_val: np.ndarray,
    base_test: np.ndarray,
    prev_test: np.ndarray,
    q_test: np.ndarray,
    hq_test: np.ndarray,
) -> dict:
    candidates = []
    for residual_name, val_residual, test_residual in [
        ("quantum", q_val, q_test),
        ("classical_quantum", hq_val, hq_test),
    ]:
        for alpha in np.linspace(0.0, 0.8, 17):
            for gamma in np.linspace(-0.5, 0.5, 21):
                val_pred = base_val + alpha * prev_val + gamma * val_residual
                candidates.append(
                    {
                        "validation_MAE": mean_absolute_error(y_val, val_pred),
                        "alpha": float(alpha),
                        "gamma": float(gamma),
                        "residual_name": residual_name,
                        "test_pred": base_test + alpha * prev_test + gamma * test_residual,
                    }
                )
    best = min(candidates, key=lambda row: row["validation_MAE"])
    best["notes"] = (
        "rolling one-step model selected on validation; "
        f"previous_residual_alpha={best['alpha']:.3f}, quantum_residual_gamma={best['gamma']:.3f}, "
        f"residual_source={best['residual_name']}, validation_MAE={best['validation_MAE']:.6f}"
    )
    return best


def plot_metrics(metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("MAE", ascending=True)
    colors = ["#2a9d8f" if value else "#4e79a7" for value in ordered["beats_baseline"]]
    plt.figure(figsize=(11, 5))
    plt.barh(ordered["model"][::-1], ordered["MAE"].to_numpy()[::-1], color=colors[::-1])
    plt.xlabel("MAE lower is better")
    plt.title("Hybrid Quantum Postprocess vs XGBoost Baseline")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hybrid_quantum_postprocess_mae.png", dpi=150)
    plt.close()


def plot_predictions(index: pd.Index, actual: pd.Series, baseline: np.ndarray, selected: np.ndarray) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(index, actual, label="Actual", linewidth=1)
    plt.plot(index, baseline, label="XGBoost baseline", linewidth=1)
    plt.plot(index, selected, label="Hybrid quantum selected", linewidth=1)
    plt.title("Hybrid Quantum Sequential Postprocess")
    plt.xlabel("Datetime")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hybrid_quantum_postprocess_predictions.png", dpi=150)
    plt.close()


def log_message(log_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"{timestamp} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def serializable_args(args: argparse.Namespace) -> dict:
    data = vars(args).copy()
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


if __name__ == "__main__":
    run_hybrid_quantum_postprocess(parse_args())
